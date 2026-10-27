"""
route_planner.py - Generates static forwarding tables from network topologies.

This module provides functions to compile hop-by-hop forwarding tables
(RoutingTable) for any source router in a NetworkTopology:
1. build_routing_table(topology, source_node):
   Computes unweighted shortest-hop routes using Breadth-First Search (BFS).
   Time complexity: O(V + E).
2. build_weighted_routing_table(topology, source_node):
   Computes minimum-total-cost routes using Dijkstra's shortest-path algorithm.
   Time complexity: O((V + E) log V).

Forwarding principle:
Each forwarding entry maps a reachable destination node identifier to the
(host, port) network address of the IMMEDIATE NEXT HOP on the computed path
from the source to that destination. Packets destined for a multi-hop node
are forwarded to the next router along the path, which will in turn forward
the packet toward its ultimate destination.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from NetworkPacketRouting.routing_table import RoutingTable
from NetworkPacketRouting.topology import (
    InvalidNodeError,
    NetworkTopology,
    NodeNotFoundError,
    TopologyError,
    _validate_node_id,
)


class RoutePlannerError(Exception):
    """Base exception for route planning operations."""


def build_routing_table(
    topology: NetworkTopology, source_node: str
) -> RoutingTable:
    """
    Constructs a forwarding table for source_node across the network topology.

    Computes unweighted shortest-hop routes from source_node to all reachable
    destinations using a single-pass Breadth-First Search (BFS). For each
    reachable destination, the routing entry specifies the (host, port) of the
    immediate next hop router.

    Args:
        topology: The NetworkTopology containing nodes, endpoints, and links.
        source_node: The identifier of the node for which the routing table
                     is being compiled.

    Returns:
        A RoutingTable populated with entries for all reachable destinations.
        The table is initialized with populate_default=False (no synthetic
        default routes). Unreachable destinations and the source node itself
        are omitted.

    Raises:
        RoutePlannerError: If topology is not an instance of NetworkTopology.
        InvalidNodeError: If source_node identifier is invalid.
        NodeNotFoundError: If source_node is not registered in the topology.
    """
    if not isinstance(topology, NetworkTopology):
        raise RoutePlannerError(
            f"Expected a NetworkTopology instance, got {type(topology).__name__}"
        )

    valid_source = _validate_node_id(source_node)
    if not topology.has_node(valid_source):
        raise NodeNotFoundError(
            f"Source node {valid_source!r} is not registered in the topology"
        )

    routing_table = RoutingTable(populate_default=False)

    # first_hop maps: reachable_destination -> immediate_next_hop_from_source
    first_hop: Dict[str, str] = {}
    visited: Set[str] = {valid_source}
    queue: deque[str] = deque()

    # Step 1: Initialize BFS with direct neighbors of the source node.
    # For every direct neighbor, its next hop from source is itself.
    for neighbor in topology.neighbors(valid_source):
        visited.add(neighbor)
        first_hop[neighbor] = neighbor
        queue.append(neighbor)

    # Step 2: Explore remaining reachable nodes.
    # Any node discovered inherits the first_hop of the branch that found it.
    while queue:
        current = queue.popleft()
        current_first_hop = first_hop[current]

        for neighbor in topology.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                first_hop[neighbor] = current_first_hop
                queue.append(neighbor)

    # Step 3: Populate routing table entries using next-hop endpoints.
    # Note: destination == valid_source is never added to first_hop, so the
    # source itself is never given a route in its own forwarding table.
    for destination, next_hop_node in first_hop.items():
        next_hop_host, next_hop_port = topology.get_endpoint(next_hop_node)
        routing_table.add_route(destination, next_hop_host, next_hop_port)

    return routing_table


def build_weighted_routing_table(
    topology: NetworkTopology, source_node: str
) -> RoutingTable:
    """
    Constructs a forwarding table for source_node across the network topology
    using Dijkstra's shortest-path algorithm.

    Computes least-cost paths from source_node to all reachable destinations by
    running Dijkstra's algorithm once. Next hops are derived in a linear-time
    O(V) traversal of the resulting predecessor tree:
    - Direct children of the source in the shortest-path tree receive themselves
      as their next hop.
    - Deeper descendants inherit the next hop of their ancestor branch off the source.
    - Each forwarding entry in the returned RoutingTable maps a reachable
      destination to the (host, port) network address of the IMMEDIATE FIRST HOP.
    - The source node itself and any unreachable destinations are excluded.

    Time complexity: O((V + E) log V) where V is the number of registered nodes
    and E is the number of undirected links in the topology.
    Space complexity: O(V + E) auxiliary space for graph structures, distances,
    predecessors, and routing entries.

    Args:
        topology: The NetworkTopology containing nodes, endpoints, and links.
        source_node: The identifier of the node for which the routing table
                     is being compiled.

    Returns:
        A RoutingTable populated with entries for all reachable destinations.
        The table is initialized with populate_default=False. Unreachable
        destinations and the source node itself are omitted.

    Raises:
        RoutePlannerError: If topology is not an instance of NetworkTopology.
        InvalidNodeError: If source_node identifier is invalid.
        NodeNotFoundError: If source_node is not registered in the topology.
        DistanceOverflowError: If floating-point distance accumulation overflows to infinity.
    """
    if not isinstance(topology, NetworkTopology):
        raise RoutePlannerError(
            f"Expected a NetworkTopology instance, got {type(topology).__name__}"
        )

    valid_source = _validate_node_id(source_node)
    if not topology.has_node(valid_source):
        raise NodeNotFoundError(
            f"Source node {valid_source!r} is not registered in the topology"
        )

    # Obtain defensive copy of the underlying NetworkGraph using public topology API
    graph = topology.get_graph()

    # Step 1: Execute Dijkstra once from the source node
    distances, predecessors = graph.dijkstra(valid_source)

    # Step 2: Build predecessor-tree child relationships: parent -> list of children
    children: Dict[str, List[str]] = {}
    for child, parent in predecessors.items():
        children.setdefault(parent, []).append(child)

    # Step 3: Linear-time O(V) traversal to propagate first hops down the tree
    # first_hop maps: reachable_destination -> immediate_first_hop_from_source
    first_hop: Dict[str, str] = {}
    queue: deque[Tuple[str, str]] = deque()

    # For direct children of the source, the first hop is that child itself
    for direct_child in children.get(valid_source, []):
        first_hop[direct_child] = direct_child
        queue.append((direct_child, direct_child))

    # Deeper descendants inherit the first hop of their ancestor branch
    while queue:
        current_node, branch_hop = queue.popleft()
        for child in children.get(current_node, []):
            first_hop[child] = branch_hop
            queue.append((child, branch_hop))

    # Step 4: Populate routing table entries with first-hop endpoints
    routing_table = RoutingTable(populate_default=False)
    for destination, next_hop_node in first_hop.items():
        next_hop_host, next_hop_port = topology.get_endpoint(next_hop_node)
        routing_table.add_route(destination, next_hop_host, next_hop_port)

    return routing_table
