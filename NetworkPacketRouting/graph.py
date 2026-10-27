"""
graph.py - Custom adjacency-list graph with BFS and DFS traversals.

Implements an undirected, unweighted graph for network topology modeling.
All algorithms operate on adjacency lists for O(V + E) time complexity.

Design decisions:
- Nodes are string identifiers representing routers / network endpoints.
- Edges are undirected: add_edge(a, b) creates both a->b and b->a adjacency.
- add_edge() automatically creates missing endpoint nodes (documented and tested).
- Duplicate edge additions are silently ignored (no duplicate neighbors).
- Neighbor visitation order is deterministic: edge insertion order
  (Python 3.7+ dict key ordering guarantee).
- Removing a nonexistent node or edge is a safe no-op returning False.
- Internal adjacency lists are never exposed; neighbors() returns a copy.

Complexity (all expected / amortized):
- Storage: O(V + E) where V = vertices, E = edges.
- BFS / DFS traversal: O(V + E). Neighbors are iterated in insertion
  order without sorting, so no O(deg * log deg) overhead per vertex.
- add_node: O(1).
- add_edge: O(1) (duplicate check via dict membership).
- remove_node: O(deg(v)) where deg(v) = degree of the node.
- remove_edge: O(1) (dict delete).
- has_node: O(1).
- neighbors: O(deg(v)) for the list copy.
"""

from __future__ import annotations

from collections import deque
import heapq
import math
from typing import Dict, List, Optional, Set, Tuple, Union

DEFAULT_WEIGHT: int = 1


class GraphError(Exception):
    """Base exception for graph operations."""


class InvalidNodeError(GraphError):
    """Raised when a node identifier is invalid."""


class NodeNotFoundError(GraphError):
    """Raised when an operation references a node that does not exist in the graph."""


class EdgeNotFoundError(GraphError):
    """Raised when an operation targets an edge that does not exist in the graph."""


class InvalidWeightError(GraphError):
    """Raised when an edge weight is invalid (e.g. negative, non-numeric, NaN, inf, or bool)."""


class DistanceOverflowError(GraphError, OverflowError):
    """Raised when floating-point path distance calculation overflows to infinity."""


def _validate_node_id(node: object) -> str:
    """
    Validates a node identifier.

    Rules:
    - Must be a string (bool rejected as non-string).
    - Must not be empty.
    - Must not be whitespace-only.
    - Must not have leading or trailing whitespace.

    Returns:
        The validated node string.

    Raises:
        InvalidNodeError: If the identifier violates any rule.
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


def _validate_weight(weight: object) -> Union[int, float]:
    """
    Validates that an edge weight is a non-negative, finite real number (int or float).

    Rules:
    - Must be an int or float.
    - Booleans are rejected (even though bool is a subclass of int in Python).
    - If float, must not be NaN or infinite (positive or negative infinity).
    - If int, arbitrary-precision integers are supported and cannot be NaN/infinite.
    - Must be non-negative (>= 0). Zero is explicitly permitted.

    Returns:
        The validated numeric weight (int or float).

    Raises:
        InvalidWeightError: If weight fails any validation check.
    """
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise InvalidWeightError(
            f"Edge weight must be a numeric int or float, got {type(weight).__name__}"
        )
    if isinstance(weight, float):
        if math.isnan(weight):
            raise InvalidWeightError("Edge weight cannot be NaN")
        if math.isinf(weight):
            raise InvalidWeightError("Edge weight cannot be infinite")
    if weight < 0:
        raise InvalidWeightError(f"Edge weight must be non-negative, got {weight}")
    return weight


class NetworkGraph:
    """
    Undirected, optionally weighted graph represented as adjacency lists.

    Each node is a string identifier (e.g. a router name). Edges connect
    two nodes symmetrically with an associated non-negative numeric weight
    (defaulting to 1). Internal storage uses ``dict[str, dict[str, Union[int, float]]]``
    for O(1) edge-existence and weight lookups with guaranteed insertion-order
    iteration (Python 3.7+), providing deterministic traversal without sorting.

    Example::

        g = NetworkGraph()
        g.add_edge("A", "B", weight=2.5)
        g.add_edge("A", "C")  # defaults to weight 1
        assert g.neighbors("A") == ["B", "C"]  # insertion order
        assert g.get_edge_weight("A", "B") == 2.5
        assert g.bfs("A") == ["A", "B", "C"]
    """

    def __init__(self) -> None:
        """Initializes an empty graph with no nodes or edges."""
        self._adjacency: Dict[str, Dict[str, Union[int, float]]] = {}

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: str) -> bool:
        """
        Adds a node to the graph if it does not already exist.

        Args:
            node: A valid string identifier for the node.

        Returns:
            True if the node was newly added, False if it already existed.

        Raises:
            InvalidNodeError: If the node identifier is invalid.
        """
        node = _validate_node_id(node)
        if node in self._adjacency:
            return False
        self._adjacency[node] = {}
        return True

    def remove_node(self, node: str) -> bool:
        """
        Removes a node and all its incident edges from the graph.

        Safe to call with a nonexistent node; returns False without error.

        Args:
            node: The node identifier to remove.

        Returns:
            True if the node existed and was removed, False otherwise.

        Raises:
            InvalidNodeError: If the node identifier is invalid.
        """
        node = _validate_node_id(node)
        if node not in self._adjacency:
            return False
        # Remove this node from all neighbors' adjacency sets
        for neighbor in self._adjacency[node]:
            del self._adjacency[neighbor][node]
        del self._adjacency[node]
        return True

    def has_node(self, node: str) -> bool:
        """
        Checks whether a node exists in the graph.

        Args:
            node: The node identifier to look up.

        Returns:
            True if the node exists, False otherwise.

        Raises:
            InvalidNodeError: If the node identifier is invalid.
        """
        node = _validate_node_id(node)
        return node in self._adjacency

    def nodes(self) -> List[str]:
        """Returns a sorted list of all node identifiers in the graph."""
        return sorted(self._adjacency.keys())

    def node_count(self) -> int:
        """Returns the number of nodes in the graph."""
        return len(self._adjacency)

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(
        self,
        node_a: str,
        node_b: str,
        weight: Union[int, float] = DEFAULT_WEIGHT,
    ) -> bool:
        """
        Adds an undirected edge between two nodes with an optional weight.

        If either endpoint node does not exist, it is automatically created.
        Self-loops (node_a == node_b) are rejected.
        Duplicate edge additions are silently ignored and return False without
        mutating the existing edge or its weight.
        Because the graph is undirected, the same weight applies symmetrically
        in both directions: (node_a, node_b) and (node_b, node_a).

        Time complexity: O(1).
        Space complexity: O(1) amortized.

        Args:
            node_a: First endpoint.
            node_b: Second endpoint.
            weight: Non-negative numeric weight (int or float). Defaults to 1.

        Returns:
            True if the edge was newly created, False if it already existed.

        Raises:
            InvalidNodeError: If either node identifier is invalid.
            InvalidWeightError: If weight is negative, non-numeric, NaN, inf, or bool.
            GraphError: If a self-loop is attempted.
        """
        node_a = _validate_node_id(node_a)
        node_b = _validate_node_id(node_b)
        valid_weight = _validate_weight(weight)
        if node_a == node_b:
            raise GraphError(
                f"Self-loops are not allowed: {node_a!r}"
            )
        # Auto-create missing nodes
        if node_a not in self._adjacency:
            self._adjacency[node_a] = {}
        if node_b not in self._adjacency:
            self._adjacency[node_b] = {}
        # Check for duplicate
        if node_b in self._adjacency[node_a]:
            return False
        self._adjacency[node_a][node_b] = valid_weight
        self._adjacency[node_b][node_a] = valid_weight
        return True

    def remove_edge(self, node_a: str, node_b: str) -> bool:
        """
        Removes the undirected edge between two nodes.

        Safe to call when the edge or either node does not exist;
        returns False without error.

        Args:
            node_a: First endpoint.
            node_b: Second endpoint.

        Returns:
            True if the edge existed and was removed, False otherwise.

        Raises:
            InvalidNodeError: If either node identifier is invalid.
        """
        node_a = _validate_node_id(node_a)
        node_b = _validate_node_id(node_b)
        if node_a not in self._adjacency or node_b not in self._adjacency:
            return False
        if node_b not in self._adjacency[node_a]:
            return False
        del self._adjacency[node_a][node_b]
        del self._adjacency[node_b][node_a]
        return True

    def has_edge(self, node_a: str, node_b: str) -> bool:
        """
        Checks whether an edge exists between two nodes.

        Args:
            node_a: First endpoint.
            node_b: Second endpoint.

        Returns:
            True if the edge exists, False otherwise.

        Raises:
            InvalidNodeError: If either node identifier is invalid.
        """
        node_a = _validate_node_id(node_a)
        node_b = _validate_node_id(node_b)
        if node_a not in self._adjacency:
            return False
        return node_b in self._adjacency[node_a]

    def get_edge_weight(self, node_a: str, node_b: str) -> Union[int, float]:
        """
        Retrieves the numeric weight of an undirected edge.

        Because the graph is undirected, get_edge_weight(u, v) == get_edge_weight(v, u).

        Time complexity: O(1).

        Args:
            node_a: First endpoint of the edge.
            node_b: Second endpoint of the edge.

        Returns:
            The non-negative numeric weight of the edge (int or float).

        Raises:
            InvalidNodeError: If either node identifier is invalid.
            NodeNotFoundError: If either node does not exist in the graph.
            EdgeNotFoundError: If no edge exists between node_a and node_b.
        """
        node_a = _validate_node_id(node_a)
        node_b = _validate_node_id(node_b)

        if node_a not in self._adjacency:
            raise NodeNotFoundError(f"Node {node_a!r} does not exist in the graph")
        if node_b not in self._adjacency:
            raise NodeNotFoundError(f"Node {node_b!r} does not exist in the graph")

        if node_b not in self._adjacency[node_a]:
            raise EdgeNotFoundError(
                f"Edge ({node_a!r}, {node_b!r}) does not exist in the graph"
            )

        return self._adjacency[node_a][node_b]

    def edge_weight(self, node_a: str, node_b: str) -> Union[int, float]:
        """Alias for get_edge_weight(node_a, node_b)."""
        return self.get_edge_weight(node_a, node_b)

    def set_edge_weight(
        self, node_a: str, node_b: str, weight: Union[int, float]
    ) -> None:
        """
        Updates the weight of an existing undirected edge in both directions.

        Preserves existing neighbor insertion order without re-inserting keys.

        Time complexity: O(1).

        Args:
            node_a: First endpoint of the edge.
            node_b: Second endpoint of the edge.
            weight: New non-negative numeric weight (int or float).

        Raises:
            InvalidNodeError: If either node identifier is invalid.
            InvalidWeightError: If weight is negative, non-numeric, NaN, inf, or bool.
            NodeNotFoundError: If either node does not exist in the graph.
            EdgeNotFoundError: If no edge exists between node_a and node_b.
        """
        node_a = _validate_node_id(node_a)
        node_b = _validate_node_id(node_b)
        valid_weight = _validate_weight(weight)

        if node_a not in self._adjacency:
            raise NodeNotFoundError(f"Node {node_a!r} does not exist in the graph")
        if node_b not in self._adjacency:
            raise NodeNotFoundError(f"Node {node_b!r} does not exist in the graph")

        if node_b not in self._adjacency[node_a]:
            raise EdgeNotFoundError(
                f"Edge ({node_a!r}, {node_b!r}) does not exist in the graph"
            )

        self._adjacency[node_a][node_b] = valid_weight
        self._adjacency[node_b][node_a] = valid_weight

    def update_edge_weight(
        self, node_a: str, node_b: str, weight: Union[int, float]
    ) -> None:
        """Alias for set_edge_weight(node_a, node_b, weight)."""
        self.set_edge_weight(node_a, node_b, weight)

    def edge_count(self) -> int:
        """
        Returns the number of undirected edges in the graph.

        Each undirected edge is counted once (not twice).
        """
        return sum(len(neighbors) for neighbors in self._adjacency.values()) // 2

    def neighbors(self, node: str) -> List[str]:
        """
        Returns a list of neighbors in edge insertion order.

        The returned list is a copy; mutating it does not affect the graph.
        Insertion order (Python 3.7+ dict guarantee) ensures deterministic
        traversal for BFS/DFS without sorting overhead.

        Args:
            node: The node whose neighbors to retrieve.

        Returns:
            A list of neighbor node identifiers in insertion order.

        Raises:
            InvalidNodeError: If the node identifier is invalid.
            NodeNotFoundError: If the node does not exist in the graph.
        """
        node = _validate_node_id(node)
        if node not in self._adjacency:
            raise NodeNotFoundError(f"Node {node!r} does not exist in the graph")
        return list(self._adjacency[node].keys())

    # ------------------------------------------------------------------
    # Traversal algorithms
    # ------------------------------------------------------------------

    def bfs(self, start: str) -> List[str]:
        """
        Breadth-First Search: returns nodes in BFS visitation order.

        Uses a FIFO queue (collections.deque). Neighbors are visited in
        edge insertion order for deterministic results.

        Time complexity: O(V + E).
        Space complexity: O(V) for the visited set and queue.

        Args:
            start: The node to begin traversal from.

        Returns:
            A list of node identifiers in the order they were first visited.

        Raises:
            InvalidNodeError: If the start identifier is invalid.
            NodeNotFoundError: If the start node does not exist in the graph.
        """
        start = _validate_node_id(start)
        if start not in self._adjacency:
            raise NodeNotFoundError(
                f"BFS start node {start!r} does not exist in the graph"
            )

        visited: Set[str] = {start}
        queue: deque[str] = deque([start])
        order: List[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbor in self._adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return order

    def dfs(self, start: str) -> List[str]:
        """
        Depth-First Search: returns nodes in DFS visitation order.

        Uses an explicit LIFO stack (list) to avoid Python's recursion-depth
        limits. Neighbors are pushed in reverse insertion order so that the
        first-inserted neighbor is visited first (popped from top of stack).

        Time complexity: O(V + E).
        Space complexity: O(V) for the visited set and stack.

        Args:
            start: The node to begin traversal from.

        Returns:
            A list of node identifiers in the order they were first visited.

        Raises:
            InvalidNodeError: If the start identifier is invalid.
            NodeNotFoundError: If the start node does not exist in the graph.
        """
        start = _validate_node_id(start)
        if start not in self._adjacency:
            raise NodeNotFoundError(
                f"DFS start node {start!r} does not exist in the graph"
            )

        visited: Set[str] = set()
        stack: List[str] = [start]
        order: List[str] = []

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            order.append(current)
            # Push neighbors in reverse insertion order so first-inserted is on top
            for neighbor in reversed(list(self._adjacency[current])):
                if neighbor not in visited:
                    stack.append(neighbor)

        return order

    def bfs_tree(self, start: str) -> Dict[str, Optional[str]]:
        """
        BFS traversal tree: returns a parent mapping.

        The start node maps to None. Every other reachable node maps to
        the node from which it was first discovered during BFS.

        Time complexity: O(V + E).

        Args:
            start: The root node for the BFS tree.

        Returns:
            A dict mapping each visited node to its parent (None for start).

        Raises:
            InvalidNodeError: If the start identifier is invalid.
            NodeNotFoundError: If the start node does not exist in the graph.
        """
        start = _validate_node_id(start)
        if start not in self._adjacency:
            raise NodeNotFoundError(
                f"BFS tree start node {start!r} does not exist in the graph"
            )

        parent: Dict[str, Optional[str]] = {start: None}
        queue: deque[str] = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor in self._adjacency[current]:
                if neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)

        return parent

    def dfs_tree(self, start: str) -> Dict[str, Optional[str]]:
        """
        DFS traversal tree: returns a parent mapping.

        The start node maps to None. Every other reachable node maps to
        the node from which it was first discovered during DFS (using an
        explicit stack).

        Time complexity: O(V + E).

        Args:
            start: The root node for the DFS tree.

        Returns:
            A dict mapping each visited node to its parent (None for start).

        Raises:
            InvalidNodeError: If the start identifier is invalid.
            NodeNotFoundError: If the start node does not exist in the graph.
        """
        start = _validate_node_id(start)
        if start not in self._adjacency:
            raise NodeNotFoundError(
                f"DFS tree start node {start!r} does not exist in the graph"
            )

        parent: Dict[str, Optional[str]] = {start: None}
        # Stack entries: (node_to_visit, parent_of_that_node)
        stack: List[tuple] = [(start, None)]

        while stack:
            current, current_parent = stack.pop()
            if current in parent and current != start:
                continue
            if current != start:
                parent[current] = current_parent

            for neighbor in reversed(list(self._adjacency[current])):
                if neighbor not in parent:
                    stack.append((neighbor, current))

        return parent

    # ------------------------------------------------------------------
    # Weighted Shortest Path Algorithms (Dijkstra)
    # ------------------------------------------------------------------

    def dijkstra(
        self, source: str
    ) -> Tuple[Dict[str, Union[int, float]], Dict[str, str]]:
        """
        Computes shortest paths from source to all nodes using Dijkstra's algorithm.

        Uses a min-priority queue (heapq) with deterministic tie-breaking and
        efficient skipping of stale heap entries.
        Supports both integer and floating-point edge weights (including zero weights).
        Preserves exact arbitrary-precision integer arithmetic for integer-only paths.

        Time complexity: O((V + E) log V) where V = nodes, E = edges.
        Space complexity: O(V + E) for priority queue, distance and predecessor maps.

        Args:
            source: The starting node identifier.

        Returns:
            A tuple of (distances, predecessors):
            - distances: Dict[str, Union[int, float]] mapping EVERY node in the graph
              to its shortest-path distance from source. Source distance is 0.
              Unreachable nodes map to math.inf.
            - predecessors: Dict[str, str] mapping each reachable non-source node
              to its predecessor along the shortest path. The source and unreachable
              nodes have no predecessor and are not keys in this dictionary.

        Raises:
            InvalidNodeError: If source identifier is invalid.
            NodeNotFoundError: If source does not exist in the graph.
            DistanceOverflowError: If floating-point distance accumulation overflows to infinity.
        """
        source = _validate_node_id(source)
        if source not in self._adjacency:
            raise NodeNotFoundError(f"Source node {source!r} does not exist in the graph")

        # Initialize distances: source is 0 (exact integer), all others infinity
        distances: Dict[str, Union[int, float]] = {
            node: math.inf for node in self._adjacency
        }
        distances[source] = 0

        predecessors: Dict[str, str] = {}
        visited: Set[str] = set()

        # Priority queue entries: (distance, insertion_counter, node)
        # insertion_counter guarantees deterministic FIFO tie-breaking for equal distances
        # and avoids comparing node names or unorderable types.
        entry_counter = 0
        heap: List[Tuple[Union[int, float], int, str]] = [(0, entry_counter, source)]

        while heap:
            curr_dist, _, current = heapq.heappop(heap)

            # Skip stale heap entries
            if current in visited:
                continue
            visited.add(current)

            for neighbor, weight in self._adjacency[current].items():
                if neighbor in visited:
                    continue

                new_dist = curr_dist + weight
                if isinstance(new_dist, float) and math.isinf(new_dist):
                    raise DistanceOverflowError(
                        f"Floating-point distance overflowed to infinity between "
                        f"{current!r} and {neighbor!r}"
                    )

                # Strict distance improvement: preserves deterministic tie-breaking
                # and prevents predecessor cycles on zero-weight edges.
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current
                    entry_counter += 1
                    heapq.heappush(heap, (new_dist, entry_counter, neighbor))

        return distances, predecessors

    def shortest_path_weighted(
        self, source: str, target: str
    ) -> Optional[Tuple[List[str], Union[int, float]]]:
        """
        Computes the shortest weighted path and its total cost between source and target.

        Reconstructs the sequence of nodes along the minimum-weight path using
        Dijkstra's algorithm.

        Time complexity: O((V + E) log V).
        Space complexity: O(V + E).

        Args:
            source: The starting node identifier.
            target: The destination node identifier.

        Returns:
            A tuple of (path, total_cost) if target is reachable, where:
            - path: List[str] of node identifiers from source to target inclusive.
            - total_cost: The non-negative numeric cost (int or float) of the path.
            Returns ([source], 0) if source == target.
            Returns None if target is unreachable from source.

        Raises:
            InvalidNodeError: If source or target identifier is invalid.
            NodeNotFoundError: If source or target does not exist in the graph.
            DistanceOverflowError: If floating-point distance accumulation overflows to infinity.
        """
        source = _validate_node_id(source)
        target = _validate_node_id(target)

        if source not in self._adjacency:
            raise NodeNotFoundError(f"Source node {source!r} does not exist in the graph")
        if target not in self._adjacency:
            raise NodeNotFoundError(f"Target node {target!r} does not exist in the graph")

        if source == target:
            return ([source], 0)

        distances, predecessors = self.dijkstra(source)

        if target not in predecessors:
            return None

        # Reconstruct path from target back to source
        path: List[str] = []
        curr: Optional[str] = target
        visited_nodes: Set[str] = set()

        while curr is not None:
            if curr in visited_nodes:
                raise GraphError(
                    f"Predecessor cycle detected at node {curr!r} during path reconstruction"
                )
            visited_nodes.add(curr)
            path.append(curr)
            if curr == source:
                break
            curr = predecessors.get(curr)

        path.reverse()
        return (path, distances[target])
