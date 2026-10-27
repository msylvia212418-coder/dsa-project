"""
topology_router.py - Factory for instantiating Routers from NetworkTopology.

This module provides the create_topology_router() factory function. It bridges
the graph-based NetworkTopology model with the physical TCP/IP Router engine
developed in Milestone 1.

Usage & Scope:
- The factory is intended for nodes that act as routing/forwarding switches
  within the network mesh.
- Leaf/destination-only nodes that exclusively consume incoming packets
  (e.g., receiver-1) should run a dedicated TCP receiver process rather than
  a Router.
- The returned Router is configured with the node's registered host and port,
  and its forwarding table is compiled via build_routing_table().
"""

from __future__ import annotations

from typing import Optional

from NetworkPacketRouting.route_planner import (
    build_routing_table,
    build_weighted_routing_table,
)
from NetworkPacketRouting.router import DOWNSTREAM_TIMEOUT, Router
from NetworkPacketRouting.topology import (
    InvalidNodeError,
    NetworkTopology,
    NodeNotFoundError,
)


class TopologyRouterError(Exception):
    """Base exception for topology-router factory errors."""


class InvalidRoutingModeError(TopologyRouterError, ValueError):
    """Raised when an unsupported routing mode is specified."""


def create_topology_router(
    topology: NetworkTopology,
    node_id: str,
    downstream_timeout: float = DOWNSTREAM_TIMEOUT,
    routing_mode: str = "unweighted",
) -> Router:
    """
    Creates and configures a Router instance for a specific node in a topology.

    Validates that node_id exists in the topology, queries its registered
    (host, port) endpoint, generates its forwarding table using the chosen
    routing mode ('unweighted' or 'weighted'), and returns an unstarted Router instance.

    Note:
        This factory is for nodes participating as routing switches. Nodes
        acting solely as packet endpoints/consumers should run a receiver
        service instead.

    Args:
        topology: The NetworkTopology containing nodes, endpoints, and links.
        node_id: The identifier of the node to instantiate as a Router.
        downstream_timeout: Connection and forwarding timeout in seconds
                            when transmitting packets to downstream hops.
                            Defaults to router.DOWNSTREAM_TIMEOUT.
        routing_mode: Pathfinding strategy used to build the router's initial
                      forwarding table. Supported values are:
                      - 'unweighted': BFS shortest-hop pathfinding (default).
                      - 'weighted': Dijkstra least-cost pathfinding.

    Returns:
        A configured, unstarted Router instance. Call router.start() and
        router.serve_forever() to run the router process.

    Raises:
        TopologyRouterError: If topology is not a NetworkTopology instance.
        InvalidNodeError: If node_id is not a valid string identifier.
        NodeNotFoundError: If node_id is not registered in the topology.
        InvalidRoutingModeError: If routing_mode is not 'unweighted' or 'weighted'.
    """
    if not isinstance(topology, NetworkTopology):
        raise TopologyRouterError(
            f"Expected a NetworkTopology instance, got {type(topology).__name__}"
        )

    if not topology.has_node(node_id):
        raise NodeNotFoundError(
            f"Node {node_id!r} is not registered in the topology"
        )

    if routing_mode == "unweighted":
        routing_table = build_routing_table(topology, node_id)
    elif routing_mode == "weighted":
        routing_table = build_weighted_routing_table(topology, node_id)
    else:
        raise InvalidRoutingModeError(
            f"Unsupported routing mode {routing_mode!r}. "
            f"Supported modes are: 'unweighted', 'weighted'"
        )

    host, port = topology.get_endpoint(node_id)

    return Router(
        host=host,
        port=port,
        routing_table=routing_table,
        downstream_timeout=downstream_timeout,
    )
