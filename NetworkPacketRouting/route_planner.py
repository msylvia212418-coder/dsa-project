"""
route_planner.py - Generates static forwarding tables from network topologies.

This module provides the build_routing_table() function, which compiles a
hop-by-hop forwarding table (RoutingTable) for any source router in a
NetworkTopology using Breadth-First Search (BFS).

Forwarding principle:
Each forwarding entry maps a reachable destination node identifier to the
(host, port) network address of the IMMEDIATE NEXT HOP on the shortest
unweighted path from the source to that destination. Packets destined for
a multi-hop node are forwarded to the next router along the path, which will
in turn forward the packet toward its ultimate destination.

Algorithmic details & time complexity:
- A single BFS traversal is executed starting from the source node.
- Direct neighbors of the source are assigned themselves as their first hop.
- For all further reachable nodes, the first hop is inherited from the
  predecessor along the BFS discovery tree.
- Time complexity: O(V + E) where V is the number of vertices and E is the
  number of edges in the network topology. This single-pass approach avoids
  the O(V * (V + E)) overhead of running independent BFS queries for every
  destination.
- Space complexity: O(V) for the queue, visited set, and first-hop mapping.
- Tie-breaking: Strictly deterministic according to the underlying NetworkGraph's
  edge insertion order.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Set

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
