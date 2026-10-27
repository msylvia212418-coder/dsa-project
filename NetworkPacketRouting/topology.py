"""
topology.py - Network topology representation, unweighted BFS, and weighted pathfinding.

This module provides the NetworkTopology class, which models an interconnected
network of routers/nodes. It composes a NetworkGraph for topological structure
and maintains a registry of network endpoints (host, port).

Pathfinding algorithms:
- shortest_path(source, destination): Computes the minimum-hop path using
  Breadth-First Search (BFS).
- next_hop(source, destination): Determines the immediate forwarding hop
  along the computed unweighted shortest path.
- shortest_path_weighted(source, destination): Computes the minimum-cost
  path and total numeric weight using Dijkstra's algorithm.

Tie-breaking policy:
- For unweighted BFS (shortest_path): When multiple paths have the exact same
  minimum hop count, traversal order is strictly deterministic and follows the
  edge insertion order of the underlying NetworkGraph.
- For Dijkstra (shortest_path_weighted): When multiple paths have identical minimum
  costs, ties are broken deterministically as implemented in NetworkGraph.dijkstra().
"""

from __future__ import annotations

from collections import deque
import copy
from typing import Dict, List, Optional, Set, Tuple, Union

from NetworkPacketRouting.graph import (
    DEFAULT_WEIGHT,
    DistanceOverflowError,
    EdgeNotFoundError,
    InvalidNodeError,
    InvalidWeightError,
    NetworkGraph,
)

MIN_PORT: int = 1
MAX_PORT: int = 65535
_UNSET: object = object()


class TopologyError(Exception):
    """Base exception for network topology operations."""


class NodeNotFoundError(TopologyError):
    """Raised when an operation references a node not registered in the topology."""


class DuplicateNodeError(TopologyError):
    """Raised when attempting to register an already-registered node without overwrite=True."""


class InvalidEndpointError(TopologyError):
    """Raised when endpoint host or port fails validation."""


class DuplicateEndpointError(TopologyError):
    """Raised when an endpoint (host, port) is already bound to another registered node."""


class LinkError(TopologyError):
    """Raised when a link operation is invalid (e.g. self-loop)."""


class LinkNotFoundError(LinkError, EdgeNotFoundError):
    """Raised when an operation targets a link that does not exist in the topology."""


def _validate_node_id(node: object) -> str:
    """
    Validates a node identifier string.

    Rules (consistent with routing_table and graph):
    - Must be a string (bool rejected).
    - Must not be empty.
    - Must not have leading or trailing whitespace.
    - Must not be whitespace-only.

    Returns:
        The validated node identifier.

    Raises:
        InvalidNodeError: If validation fails.
    """
    if type(node) is not str:
        raise InvalidNodeError(
            f"Node identifier must be a string, got {type(node).__name__}"
        )
    if not node:
        raise InvalidNodeError("Node identifier must not be empty")
    if node != node.strip():
        raise InvalidNodeError(
            f"Node identifier must not have leading or trailing whitespace: {node!r}"
        )
    if not node.strip():
        raise InvalidNodeError("Node identifier must not be whitespace-only")
    return node


def _validate_host(host: object) -> str:
    """
    Validates an endpoint host address.

    Rules (consistent with routing_table):
    - Must be a string (bool rejected).
    - Must not be empty.
    - Must not have leading or trailing whitespace.

    Returns:
        The validated host string.

    Raises:
        InvalidEndpointError: If validation fails.
    """
    if type(host) is not str:
        raise InvalidEndpointError(
            f"Host must be a string, got {type(host).__name__}"
        )
    if not host:
        raise InvalidEndpointError("Host cannot be empty")
    if host != host.strip():
        raise InvalidEndpointError(
            f"Host cannot contain leading or trailing whitespace: {host!r}"
        )
    return host


def _validate_port(port: object) -> int:
    """
    Validates an endpoint TCP port number.

    Rules (consistent with routing_table):
    - Must be an integer (bool rejected).
    - Must be in range 1-65535 inclusive.

    Returns:
        The validated integer port.

    Raises:
        InvalidEndpointError: If validation fails.
    """
    if type(port) is not int or isinstance(port, bool):
        raise InvalidEndpointError(
            f"Port must be an integer, got {type(port).__name__}"
        )
    if port < MIN_PORT or port > MAX_PORT:
        raise InvalidEndpointError(
            f"Port must be between {MIN_PORT} and {MAX_PORT}, got {port}"
        )
    return port


class NetworkTopology:
    """
    Network topology composed of a NetworkGraph and an endpoint registry.

    Responsibilities:
    - Registering nodes with their network endpoints (host, port).
    - Managing bidirectional communication links between nodes.
    - Computing unweighted shortest-hop paths using BFS.
    - Finding the next hop along the shortest path for packet routing.
    - Providing defensive copies of internal state to prevent external corruption.
    """

    def __init__(self) -> None:
        """Initializes an empty network topology with no nodes or links."""
        self._graph: NetworkGraph = NetworkGraph()
        self._endpoints: Dict[str, Tuple[str, int]] = {}

    # ------------------------------------------------------------------
    # Node and Endpoint Management
    # ------------------------------------------------------------------

    def register_node(
        self, node: str, host: str, port: int, overwrite: bool = False
    ) -> None:
        """
        Registers a node and its network endpoint in the topology.

        Args:
            node: Unique string identifier for the node.
            host: IP address or hostname for the node.
            port: TCP port (1-65535) for the node.
            overwrite: If True, allows updating an existing node's endpoint.
                       Defaults to False.

        Raises:
            InvalidNodeError: If the node identifier is invalid.
            InvalidEndpointError: If host or port validation fails.
            DuplicateNodeError: If node is already registered and overwrite=False.
            DuplicateEndpointError: If the (host, port) pair is already bound
                                   to another registered node.
        """
        valid_node = _validate_node_id(node)
        valid_host = _validate_host(host)
        valid_port = _validate_port(port)
        endpoint = (valid_host, valid_port)

        # Check for duplicate node
        if valid_node in self._endpoints and not overwrite:
            raise DuplicateNodeError(
                f"Node {valid_node!r} is already registered with endpoint "
                f"{self._endpoints[valid_node]}"
            )

        # Check for duplicate endpoint assigned to a different node
        for existing_node, existing_ep in self._endpoints.items():
            if existing_node != valid_node and existing_ep == endpoint:
                raise DuplicateEndpointError(
                    f"Endpoint {endpoint} is already assigned to node {existing_node!r}"
                )

        self._endpoints[valid_node] = endpoint
        self._graph.add_node(valid_node)

    def unregister_node(self, node: str) -> bool:
        """
        Removes a node, its endpoint, and all its incident links from the topology.

        Args:
            node: The identifier of the node to remove.

        Returns:
            True if the node was registered and removed, False otherwise.

        Raises:
            InvalidNodeError: If the node identifier is invalid.
        """
        valid_node = _validate_node_id(node)
        if valid_node not in self._endpoints:
            return False

        del self._endpoints[valid_node]
        self._graph.remove_node(valid_node)
        return True

    def has_node(self, node: str) -> bool:
        """
        Checks whether a node is registered in the topology.

        Args:
            node: The node identifier to check.

        Returns:
            True if registered, False otherwise.

        Raises:
            InvalidNodeError: If the node identifier is invalid.
        """
        valid_node = _validate_node_id(node)
        return valid_node in self._endpoints

    def node_count(self) -> int:
        """Returns the total number of registered nodes."""
        return len(self._endpoints)

    def nodes(self) -> List[str]:
        """
        Returns a list of all registered node identifiers.

        The returned list is a defensive copy.
        """
        return list(self._endpoints.keys())

    def get_endpoint(self, node: str) -> Tuple[str, int]:
        """
        Retrieves the (host, port) endpoint for a registered node.

        Args:
            node: The node identifier.

        Returns:
            A (host, port) tuple.

        Raises:
            InvalidNodeError: If the node identifier is invalid.
            NodeNotFoundError: If the node is not registered.
        """
        valid_node = _validate_node_id(node)
        if valid_node not in self._endpoints:
            raise NodeNotFoundError(
                f"Node {valid_node!r} is not registered in the topology"
            )
        return self._endpoints[valid_node]

    def get_all_endpoints(self) -> Dict[str, Tuple[str, int]]:
        """
        Returns a shallow copy of all registered node-to-endpoint mappings.

        Mutating the returned dictionary does not modify internal topology state.
        """
        return dict(self._endpoints)

    # ------------------------------------------------------------------
    # Link Management
    # ------------------------------------------------------------------

    def connect(
        self,
        node_a: str,
        node_b: str,
        weight: Union[int, float] = DEFAULT_WEIGHT,
    ) -> bool:
        """
        Creates an undirected link between two registered nodes with an optional weight.

        Args:
            node_a: First node identifier.
            node_b: Second node identifier.
            weight: Non-negative finite numeric weight (int or float).
                    Defaults to DEFAULT_WEIGHT (1).

        Returns:
            True if the link was newly created, False if it already existed.

        Raises:
            InvalidNodeError: If either node identifier is invalid.
            NodeNotFoundError: If either node is not registered in the topology.
            LinkError: If attempting to connect a node to itself (self-loop).
            InvalidWeightError: If weight is negative, boolean, non-numeric, NaN, or infinite.
        """
        valid_a = _validate_node_id(node_a)
        valid_b = _validate_node_id(node_b)

        if valid_a not in self._endpoints:
            raise NodeNotFoundError(
                f"Node {valid_a!r} is not registered in the topology"
            )
        if valid_b not in self._endpoints:
            raise NodeNotFoundError(
                f"Node {valid_b!r} is not registered in the topology"
            )
        if valid_a == valid_b:
            raise LinkError(f"Self-loops are not allowed: {valid_a!r}")

        return self._graph.add_edge(valid_a, valid_b, weight=weight)

    def add_link(
        self,
        node_a: str,
        node_b: str,
        weight: Union[int, float] = DEFAULT_WEIGHT,
    ) -> bool:
        """Alias for connect(node_a, node_b, weight=weight)."""
        return self.connect(node_a, node_b, weight=weight)

    def disconnect(self, node_a: str, node_b: str) -> bool:
        """
        Removes the undirected link between two registered nodes.

        Args:
            node_a: First node identifier.
            node_b: Second node identifier.

        Returns:
            True if the link existed and was removed, False otherwise.

        Raises:
            InvalidNodeError: If either node identifier is invalid.
            NodeNotFoundError: If either node is not registered in the topology.
        """
        valid_a = _validate_node_id(node_a)
        valid_b = _validate_node_id(node_b)

        if valid_a not in self._endpoints:
            raise NodeNotFoundError(
                f"Node {valid_a!r} is not registered in the topology"
            )
        if valid_b not in self._endpoints:
            raise NodeNotFoundError(
                f"Node {valid_b!r} is not registered in the topology"
            )

        return self._graph.remove_edge(valid_a, valid_b)

    def remove_link(self, node_a: str, node_b: str) -> bool:
        """Alias for disconnect(node_a, node_b)."""
        return self.disconnect(node_a, node_b)

    def has_link(self, node_a: str, node_b: str) -> bool:
        """
        Checks whether an undirected link exists between two nodes.

        Args:
            node_a: First node identifier.
            node_b: Second node identifier.

        Returns:
            True if the link exists, False otherwise.

        Raises:
            InvalidNodeError: If either node identifier is invalid.
        """
        valid_a = _validate_node_id(node_a)
        valid_b = _validate_node_id(node_b)

        if valid_a not in self._endpoints or valid_b not in self._endpoints:
            return False

        return self._graph.has_edge(valid_a, valid_b)

    def link_count(self) -> int:
        """Returns the total number of undirected links in the topology."""
        return self._graph.edge_count()

    def get_link_weight(self, node_a: str, node_b: str) -> Union[int, float]:
        """
        Retrieves the numeric weight of an undirected link between two registered nodes.

        Because links are undirected, get_link_weight(u, v) == get_link_weight(v, u).

        Args:
            node_a: First node identifier.
            node_b: Second node identifier.

        Returns:
            The non-negative numeric weight of the link (int or float).

        Raises:
            InvalidNodeError: If either node identifier is invalid.
            NodeNotFoundError: If either node is not registered in the topology.
            LinkNotFoundError: If no link exists between node_a and node_b.
        """
        valid_a = _validate_node_id(node_a)
        valid_b = _validate_node_id(node_b)

        if valid_a not in self._endpoints:
            raise NodeNotFoundError(
                f"Node {valid_a!r} is not registered in the topology"
            )
        if valid_b not in self._endpoints:
            raise NodeNotFoundError(
                f"Node {valid_b!r} is not registered in the topology"
            )

        if not self._graph.has_edge(valid_a, valid_b):
            raise LinkNotFoundError(
                f"Link between {valid_a!r} and {valid_b!r} does not exist in the topology"
            )

        return self._graph.get_edge_weight(valid_a, valid_b)

    def link_weight(self, node_a: str, node_b: str) -> Union[int, float]:
        """Alias for get_link_weight(node_a, node_b)."""
        return self.get_link_weight(node_a, node_b)

    def set_link_weight(
        self, node_a: str, node_b: str, weight: Union[int, float]
    ) -> None:
        """
        Updates the numeric weight of an existing undirected link.

        Args:
            node_a: First node identifier.
            node_b: Second node identifier.
            weight: Non-negative finite numeric weight (int or float).

        Raises:
            InvalidNodeError: If either node identifier is invalid.
            NodeNotFoundError: If either node is not registered in the topology.
            LinkNotFoundError: If no link exists between node_a and node_b.
            InvalidWeightError: If weight is negative, boolean, non-numeric, NaN, or infinite.
        """
        valid_a = _validate_node_id(node_a)
        valid_b = _validate_node_id(node_b)

        if valid_a not in self._endpoints:
            raise NodeNotFoundError(
                f"Node {valid_a!r} is not registered in the topology"
            )
        if valid_b not in self._endpoints:
            raise NodeNotFoundError(
                f"Node {valid_b!r} is not registered in the topology"
            )

        if not self._graph.has_edge(valid_a, valid_b):
            raise LinkNotFoundError(
                f"Link between {valid_a!r} and {valid_b!r} does not exist in the topology"
            )

        self._graph.set_edge_weight(valid_a, valid_b, weight)

    def neighbors(self, node: str) -> List[str]:
        """
        Returns a list of neighbor node identifiers in edge insertion order.

        Args:
            node: The node identifier.

        Returns:
            A list of neighbor identifiers.

        Raises:
            InvalidNodeError: If the node identifier is invalid.
            NodeNotFoundError: If the node is not registered in the topology.
        """
        valid_node = _validate_node_id(node)
        if valid_node not in self._endpoints:
            raise NodeNotFoundError(
                f"Node {valid_node!r} is not registered in the topology"
            )
        return self._graph.neighbors(valid_node)

    # ------------------------------------------------------------------
    # Graph Inspection (Defensive Copy)
    # ------------------------------------------------------------------

    def get_graph(self) -> NetworkGraph:
        """
        Returns an independent, complete defensive copy of the underlying NetworkGraph.

        Uses copy.deepcopy() to guarantee complete independence without
        accessing or mutating internal graph attributes directly.
        Mutating the returned graph has no effect on the topology's
        internal graph structure or shortest path routing.
        """
        return copy.deepcopy(self._graph)

    # ------------------------------------------------------------------
    # Pathfinding (BFS Shortest Path & Next Hop)
    # ------------------------------------------------------------------

    def shortest_path(self, source: str, destination: str) -> Optional[List[str]]:
        """
        Computes the shortest path (minimum hop count) between source and destination.

        Uses Breadth-First Search (BFS) to guarantee the minimum number of hops.
        When multiple paths have the identical shortest hop count, the path chosen
        is determined strictly by the edge insertion order of the graph: neighbors
        added earlier are explored earlier.

        Args:
            source: Starting node identifier.
            destination: Target node identifier.

        Returns:
            A list of node identifiers starting with source and ending with
            destination (e.g. ['A', 'B', 'C']).
            Returns [source] if source == destination.
            Returns None if destination is unreachable from source.

        Raises:
            InvalidNodeError: If source or destination identifier is invalid.
            NodeNotFoundError: If source or destination is not registered.
        """
        valid_source = _validate_node_id(source)
        valid_dest = _validate_node_id(destination)

        if valid_source not in self._endpoints:
            raise NodeNotFoundError(
                f"Source node {valid_source!r} is not registered in the topology"
            )
        if valid_dest not in self._endpoints:
            raise NodeNotFoundError(
                f"Destination node {valid_dest!r} is not registered in the topology"
            )

        if valid_source == valid_dest:
            return [valid_source]

        visited: Set[str] = {valid_source}
        queue: deque[str] = deque([valid_source])
        parent: Dict[str, str] = {}
        found = False

        while queue:
            current = queue.popleft()
            if current == valid_dest:
                found = True
                break

            for neighbor in self._graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    if neighbor == valid_dest:
                        found = True
                        break
                    queue.append(neighbor)

            if found:
                break

        if not found:
            return None

        # Reconstruct path from destination back to source
        path: List[str] = []
        curr: Optional[str] = valid_dest
        while curr is not None:
            path.append(curr)
            curr = parent.get(curr)

        path.reverse()
        return path

    def next_hop(self, source: str, destination: str) -> Optional[str]:
        """
        Determines the immediate next hop along the shortest path.

        Args:
            source: Starting node identifier.
            destination: Target node identifier.

        Returns:
            The identifier of the immediate next hop node.
            Returns None if source equals destination or no path exists.

        Raises:
            InvalidNodeError: If source or destination identifier is invalid.
            NodeNotFoundError: If source or destination is not registered.
        """
        path = self.shortest_path(source, destination)
        if path is None or len(path) < 2:
            return None
        return path[1]

    def shortest_path_weighted(
        self,
        source: str,
        destination: object = _UNSET,
        *,
        target: object = _UNSET,
    ) -> Optional[Tuple[List[str], Union[int, float]]]:
        """
        Computes the shortest weighted path and its total cost using Dijkstra's algorithm.

        Delegates to the underlying NetworkGraph's Dijkstra implementation.
        Validates node identifiers and verifies that both nodes are registered
        in the topology's endpoint registry.

        Args:
            source: Starting node identifier.
            destination: Target node identifier (positional or keyword).
            target: Keyword alias for destination.

        Returns:
            A tuple of (path, total_cost) if destination is reachable, where:
            - path: List[str] of node identifiers from source to destination inclusive.
            - total_cost: The non-negative numeric cost (int or float) of the path.
            Returns ([source], 0) if source == destination.
            Returns None if destination is unreachable from source.

        Raises:
            TypeError: If neither destination nor target is provided.
            InvalidNodeError: If source or destination identifier is invalid.
            NodeNotFoundError: If source or destination is not registered in the topology.
            DistanceOverflowError: If floating-point distance calculation overflows to infinity.
        """
        if destination is not _UNSET:
            dest = destination
        elif target is not _UNSET:
            dest = target
        else:
            raise TypeError(
                "shortest_path_weighted() missing 1 required positional argument: 'destination'"
            )

        valid_source = _validate_node_id(source)
        valid_dest = _validate_node_id(dest)

        if valid_source not in self._endpoints:
            raise NodeNotFoundError(
                f"Source node {valid_source!r} is not registered in the topology"
            )
        if valid_dest not in self._endpoints:
            raise NodeNotFoundError(
                f"Destination node {valid_dest!r} is not registered in the topology"
            )

        if valid_source == valid_dest:
            return ([valid_source], 0)

        return self._graph.shortest_path_weighted(valid_source, valid_dest)
