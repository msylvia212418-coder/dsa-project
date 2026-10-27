"""
resilience.py - Dynamic route refresh and link-failure recovery.

This module provides the TopologyRoutingManager, which orchestrates live
forwarding table updates across active Router instances in response to
topological changes (such as link failures or restorations).

Architectural principles:
- Router lifecycles, bound ports, and listening sockets remain uninterrupted
  during routing table refreshes.
- When a link is severed (disconnect_link), the topology is updated and
  all managed routers dynamically recompute their shortest-hop paths using BFS
  via build_routing_table().
- If a destination becomes completely unreachable, it is cleanly omitted from
  the updated forwarding table without leaving stale or broken routes.
- The manager is strictly synchronous and relies solely on Python standard-library
  mechanisms.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from NetworkPacketRouting.route_planner import build_routing_table
from NetworkPacketRouting.router import Router
from NetworkPacketRouting.routing_table import RoutingTable
from NetworkPacketRouting.topology import (
    InvalidNodeError,
    NetworkTopology,
    NodeNotFoundError,
)


class ResilienceError(Exception):
    """Base exception for resilience and topology management operations."""


class RouterNotManagedError(ResilienceError):
    """Raised when an operation targets a router node not managed by the manager."""


class InvalidManagerConfigError(ResilienceError):
    """Raised when manager configuration or arguments fail validation."""


class TopologyRoutingManager:
    """
    Coordinates dynamic routing table updates across multiple active Routers.

    Maintains a reference to a NetworkTopology and a registry of active Router
    instances. When links fail or topology changes occur, the manager recomputes
    forwarding tables and applies them in-place to the running routers.
    """

    def __init__(
        self,
        topology: NetworkTopology,
        routers: Optional[Dict[str, Router]] = None,
    ) -> None:
        """
        Initializes the TopologyRoutingManager.

        Args:
            topology: The NetworkTopology governing the network graph and endpoints.
            routers: Optional dictionary mapping node IDs to their active Router
                     instances. Each node ID must exist in the topology.

        Raises:
            InvalidManagerConfigError: If topology is not a NetworkTopology or
                                      if any router instance is invalid.
            NodeNotFoundError: If any router node ID is not registered in the topology.
            InvalidNodeError: If any router node ID is malformed.
        """
        if not isinstance(topology, NetworkTopology):
            raise InvalidManagerConfigError(
                f"Expected a NetworkTopology instance, got {type(topology).__name__}"
            )

        self._topology: NetworkTopology = topology
        self._routers: Dict[str, Router] = {}

        if routers is not None:
            if not isinstance(routers, dict):
                raise InvalidManagerConfigError(
                    f"routers must be a dict, got {type(routers).__name__}"
                )
            for node_id, router in routers.items():
                self.add_router(node_id, router)

    @property
    def topology(self) -> NetworkTopology:
        """Returns the managed NetworkTopology instance."""
        return self._topology

    def managed_nodes(self) -> List[str]:
        """Returns a list of all currently managed router node IDs."""
        return list(self._routers.keys())

    def get_router(self, node_id: str) -> Optional[Router]:
        """
        Retrieves the Router instance for a managed node.

        Args:
            node_id: The node identifier.

        Returns:
            The associated Router instance, or None if not managed.

        Raises:
            InvalidNodeError: If node_id is malformed.
        """
        if not self._topology.has_node(node_id):
            return None
        return self._routers.get(node_id)

    def add_router(self, node_id: str, router: Router) -> None:
        """
        Registers an active Router instance under the management of this manager.

        Args:
            node_id: The identifier of the node corresponding to the router.
            router: The active or unstarted Router instance.

        Raises:
            InvalidNodeError: If node_id is not a valid string identifier.
            NodeNotFoundError: If node_id is not registered in the topology.
            InvalidManagerConfigError: If router is not an instance of Router.
        """
        if not self._topology.has_node(node_id):
            raise NodeNotFoundError(
                f"Cannot manage router: Node {node_id!r} is not registered in the topology"
            )

        if not isinstance(router, Router):
            raise InvalidManagerConfigError(
                f"Expected a Router instance for node {node_id!r}, got {type(router).__name__}"
            )

        self._routers[node_id] = router

    def remove_router(self, node_id: str) -> bool:
        """
        Removes a Router from management.

        Args:
            node_id: The node identifier to remove from management.

        Returns:
            True if the router was managed and removed, False otherwise.
        """
        if node_id in self._routers:
            del self._routers[node_id]
            return True
        return False

    def refresh_router(self, node_id: str) -> RoutingTable:
        """
        Recomputes and installs an updated forwarding table for a single router.

        Args:
            node_id: The node identifier of the router to refresh.

        Returns:
            The newly compiled RoutingTable instance installed on the router.

        Raises:
            InvalidNodeError: If node_id is malformed.
            NodeNotFoundError: If node_id is not registered in the topology.
            RouterNotManagedError: If node_id has no registered Router in this manager.
            ResilienceError: If route calculation or table installation fails.
        """
        if not self._topology.has_node(node_id):
            raise NodeNotFoundError(
                f"Node {node_id!r} is not registered in the topology"
            )

        if node_id not in self._routers:
            raise RouterNotManagedError(
                f"No Router instance is managed for node {node_id!r}"
            )

        router = self._routers[node_id]
        try:
            new_table = build_routing_table(self._topology, node_id)
        except Exception as exc:
            raise ResilienceError(
                f"Failed to compute routing table for router {node_id!r}: {exc}"
            ) from exc

        try:
            router.update_routing_table(new_table)
        except Exception as exc:
            raise ResilienceError(
                f"Failed to install routing table on router {node_id!r}: {exc}"
            ) from exc

        return new_table

    def refresh_all(self) -> Dict[str, RoutingTable]:
        """
        Recomputes and installs updated forwarding tables for all managed routers.

        Executes a two-phase transactional update:
        Phase 1 (Preparation): Computes prospective RoutingTable instances for all
        managed routers into a temporary dictionary using build_routing_table().
        If computation fails for any router, a ResilienceError is raised with the
        offending router identified and the original cause chained. No router's
        existing routing table is modified (all-or-nothing preparation).
        Phase 2 (Installation): Only after all route calculations succeed, installs
        the candidate tables onto each managed router via update_routing_table().
        If an unexpected installation error occurs, a ResilienceError is raised
        documenting any partial installation state.

        Returns:
            A dictionary mapping node IDs to their newly compiled RoutingTable instances.

        Raises:
            ResilienceError: If route calculation fails for any router or if installation fails.
        """
        # Phase 1: Compute candidate routing tables into temporary mapping
        candidate_tables: Dict[str, RoutingTable] = {}
        for node_id in list(self._routers.keys()):
            if not self._topology.has_node(node_id):
                raise ResilienceError(
                    f"Managed router node {node_id!r} is not registered in the topology"
                )
            try:
                candidate_tables[node_id] = build_routing_table(self._topology, node_id)
            except Exception as exc:
                raise ResilienceError(
                    f"Failed to compute routing table for managed router {node_id!r}: {exc}"
                ) from exc

        # Phase 2: Install all candidate tables only after all calculations succeeded
        failed_installations: List[Tuple[str, Exception]] = []
        for node_id, table in candidate_tables.items():
            try:
                self._routers[node_id].update_routing_table(table)
            except Exception as exc:
                failed_installations.append((node_id, exc))

        if failed_installations:
            failed_details = ", ".join(f"{nid}: {err}" for nid, err in failed_installations)
            raise ResilienceError(
                f"Failed to install routing tables on routers: {failed_details}. "
                f"Routers may be in a partially updated state."
            )

        return candidate_tables

    def disconnect_link(self, node_a: str, node_b: str) -> bool:
        """
        Severs a communication link between two nodes and updates all routing tables.

        Hardened two-phase update strategy:
        1. Validates that both endpoints exist in the topology. If the link does not
           exist, safely returns False with zero mutations.
        2. Creates a defensive deep copy of the topology and severs the link on the
           copy to prepare prospective forwarding tables for all managed routers.
           If prospective route computation fails for any router, raises a ResilienceError
           with the original exception chained. The live topology remains intact with the
           link preserved, and no router table is altered.
        3. Once all prospective tables are successfully prepared, applies the disconnection
           to the live topology.
        4. Installs the prepared routing tables on all managed routers. If an unexpected
           installation error occurs, raises a ResilienceError detailing the partial state
           (link removed, which routers failed installation).

        Args:
            node_a: First endpoint node identifier.
            node_b: Second endpoint node identifier.

        Returns:
            True if the link existed and was removed, False if it was not present.

        Raises:
            InvalidNodeError: If either node identifier is malformed.
            NodeNotFoundError: If either node is not registered in the topology.
            ResilienceError: If prospective table calculation or installation fails.
        """
        if not self._topology.has_node(node_a):
            raise NodeNotFoundError(
                f"Node {node_a!r} is not registered in the topology"
            )
        if not self._topology.has_node(node_b):
            raise NodeNotFoundError(
                f"Node {node_b!r} is not registered in the topology"
            )

        if not self._topology.has_link(node_a, node_b):
            return False

        # Phase 1: Prepare prospective routing tables against a deep-copied topology
        candidate_topology = copy.deepcopy(self._topology)
        candidate_topology.disconnect(node_a, node_b)

        candidate_tables: Dict[str, RoutingTable] = {}
        for node_id in list(self._routers.keys()):
            if not candidate_topology.has_node(node_id):
                raise ResilienceError(
                    f"Managed router node {node_id!r} is not registered in the candidate topology"
                )
            try:
                candidate_tables[node_id] = build_routing_table(candidate_topology, node_id)
            except Exception as exc:
                raise ResilienceError(
                    f"Failed to prepare candidate routing table for managed router {node_id!r} "
                    f"during link disconnection ({node_a!r} - {node_b!r}): {exc}"
                ) from exc

        # Phase 2: Sever link on live topology (guaranteed to succeed since has_link was True)
        self._topology.disconnect(node_a, node_b)

        # Phase 3: Install prepared candidate routing tables
        failed_installations: List[Tuple[str, Exception]] = []
        for node_id, table in candidate_tables.items():
            try:
                self._routers[node_id].update_routing_table(table)
            except Exception as exc:
                failed_installations.append((node_id, exc))

        if failed_installations:
            failed_details = ", ".join(f"{nid}: {err}" for nid, err in failed_installations)
            raise ResilienceError(
                f"Topology link ({node_a!r} - {node_b!r}) was severed, but installation of updated "
                f"routing tables failed on routers: {failed_details}. The topology is disconnected "
                f"but router states may be partially updated."
            )

        return True

    def connect_link(self, node_a: str, node_b: str) -> bool:
        """
        Creates a link between two nodes and refreshes all managed router tables.

        Args:
            node_a: First endpoint node identifier.
            node_b: Second endpoint node identifier.

        Returns:
            True if the link was newly created, False if it already existed.

        Raises:
            InvalidNodeError: If either node identifier is malformed.
            NodeNotFoundError: If either node is not registered in the topology.
            LinkError: If attempting a self-loop.
            ResilienceError: If recalculation or installation on any router fails.
        """
        added = self._topology.connect(node_a, node_b)
        if added:
            self.refresh_all()
        return added
