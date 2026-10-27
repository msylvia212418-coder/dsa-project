"""
test_resilience.py - Unit and multi-hop integration tests for topology resilience.

Verifies:
- TopologyRoutingManager configuration, validation, and router registration.
- Refreshing single and all managed routers after topology alterations.
- Safe link disconnection (safe no-op for nonexistent links).
- Dynamic rerouting: removing a link updates next hops to alternative paths.
- Removal of newly-unreachable destinations from rebuilt forwarding tables.
- Multi-hop integration test over real TCP sockets:
    Initial: R1-R2, R1-R3, R3-R2, R2-receiver.
    R1's next hop is R2 (2 hops total).
    Sever R1-R2 link via disconnect_link().
    R1 automatically reroutes to R3 (R1 -> R3 -> R2 -> receiver, 3 hops total).
    Packet sent with initial TTL 8 arrives at receiver with TTL 5.
    Independent verification of packet payload, packet ID, and sender feedback.
- Clean resource teardown (sockets and threads) even if assertions fail.
"""

from __future__ import annotations

import socket
import threading
import unittest
from unittest import mock
from typing import Dict, List, Optional, Tuple

from NetworkPacketRouting.packet import Packet, RouterResponse
from NetworkPacketRouting.protocol import recv_frame, send_frame
from NetworkPacketRouting.resilience import (
    InvalidManagerConfigError,
    InvalidRoutingModeError,
    ResilienceError,
    RouterNotManagedError,
    TopologyRoutingManager,
)
from NetworkPacketRouting.route_planner import build_routing_table
from NetworkPacketRouting.router import Router
from NetworkPacketRouting.routing_table import RoutingTable
from NetworkPacketRouting.topology import (
    InvalidNodeError,
    NetworkTopology,
    NodeNotFoundError,
)
from NetworkPacketRouting.topology_router import create_topology_router


def _find_free_port() -> int:
    """Helper to allocate an available OS ephemeral port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestTopologyRoutingManagerUnit(unittest.TestCase):
    """Unit tests for TopologyRoutingManager."""

    def setUp(self):
        self.topo = NetworkTopology()
        self.topo.register_node("R1", "127.0.0.1", 5001)
        self.topo.register_node("R2", "127.0.0.1", 5002)
        self.topo.register_node("R3", "127.0.0.1", 5003)
        self.topo.connect("R1", "R2")
        self.topo.connect("R2", "R3")

        self.router_r1 = create_topology_router(self.topo, "R1")
        self.router_r2 = create_topology_router(self.topo, "R2")
        self.router_r3 = create_topology_router(self.topo, "R3")

    def test_manager_initialization_and_properties(self):
        """Verify manager initializes and registers routers."""
        mgr = TopologyRoutingManager(
            self.topo,
            {"R1": self.router_r1, "R2": self.router_r2},
        )
        self.assertIs(mgr.topology, self.topo)
        self.assertCountEqual(mgr.managed_nodes(), ["R1", "R2"])
        self.assertIs(mgr.get_router("R1"), self.router_r1)
        self.assertIs(mgr.get_router("R2"), self.router_r2)
        self.assertIsNone(mgr.get_router("R3"))

    def test_invalid_topology_raises_config_error(self):
        """Verify passing non-NetworkTopology raises InvalidManagerConfigError."""
        with self.assertRaises(InvalidManagerConfigError):
            TopologyRoutingManager("not-a-topo")  # type: ignore

    def test_unregistered_node_in_router_map_raises(self):
        """Verify registering an unmanaged node ID raises NodeNotFoundError."""
        with self.assertRaises(NodeNotFoundError):
            TopologyRoutingManager(self.topo, {"ghost": self.router_r1})

    def test_invalid_router_type_in_map_raises(self):
        """Verify non-Router value in map raises InvalidManagerConfigError."""
        with self.assertRaises(InvalidManagerConfigError):
            TopologyRoutingManager(self.topo, {"R1": "not-a-router"})  # type: ignore

    def test_add_and_remove_router(self):
        """Verify dynamically adding and removing routers."""
        mgr = TopologyRoutingManager(self.topo)
        self.assertEqual(mgr.managed_nodes(), [])

        mgr.add_router("R1", self.router_r1)
        self.assertIn("R1", mgr.managed_nodes())

        self.assertTrue(mgr.remove_router("R1"))
        self.assertFalse(mgr.remove_router("R1"))
        self.assertNotIn("R1", mgr.managed_nodes())

    def test_refresh_single_router(self):
        """Verify refresh_router updates the target router's table."""
        mgr = TopologyRoutingManager(self.topo, {"R1": self.router_r1})

        # Add a new link R1-R3 directly
        self.topo.connect("R1", "R3")

        # Before refresh: R1's table to R3 still points to R2
        self.assertEqual(
            self.router_r1.routing_table.get_route("R3"),
            ("127.0.0.1", 5002),
        )

        new_table = mgr.refresh_router("R1")
        self.assertIs(self.router_r1.routing_table, new_table)
        # After refresh: R1 has direct route to R3
        self.assertEqual(
            self.router_r1.routing_table.get_route("R3"),
            ("127.0.0.1", 5003),
        )

    def test_refresh_single_router_failure_raises_resilience_error(self):
        """Verify that a route planner failure in refresh_router raises ResilienceError."""
        mgr = TopologyRoutingManager(self.topo, {"R1": self.router_r1})
        with mock.patch(
            "NetworkPacketRouting.resilience.build_routing_table",
            side_effect=ValueError("Simulated route error"),
        ):
            with self.assertRaises(ResilienceError) as ctx:
                mgr.refresh_router("R1")

            self.assertIn("R1", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, ValueError)

    def test_refresh_unmanaged_router_raises(self):
        """Verify refreshing an unmanaged node raises RouterNotManagedError."""
        mgr = TopologyRoutingManager(self.topo, {"R1": self.router_r1})
        with self.assertRaises(RouterNotManagedError):
            mgr.refresh_router("R2")

    def test_refresh_unregistered_node_raises(self):
        """Verify refreshing an unregistered node raises NodeNotFoundError."""
        mgr = TopologyRoutingManager(self.topo)
        with self.assertRaises(NodeNotFoundError):
            mgr.refresh_router("ghost")

    def test_refresh_invalid_node_id_raises(self):
        """Verify refreshing an invalid node ID raises InvalidNodeError."""
        mgr = TopologyRoutingManager(self.topo)
        with self.assertRaises(InvalidNodeError):
            mgr.refresh_router("")

    def test_disconnect_nonexistent_link_safely_returns_false(self):
        """Verify disconnect_link on a nonexistent link returns False safely."""
        mgr = TopologyRoutingManager(self.topo, {"R1": self.router_r1})
        # R1-R3 is not connected
        self.assertFalse(mgr.disconnect_link("R1", "R3"))

    def test_disconnect_link_updates_all_managed_routers(self):
        """
        Topology:
            R1 - R2 - R3
        Disconnect R2 - R3:
            R3 becomes unreachable from R1.
            R3 must be purged from R1's routing table.
        """
        mgr = TopologyRoutingManager(
            self.topo,
            {"R1": self.router_r1, "R2": self.router_r2},
        )
        self.assertTrue(self.router_r1.routing_table.has_route("R3"))

        self.assertTrue(mgr.disconnect_link("R2", "R3"))

        # R3 is now unreachable from R1 and R2
        self.assertFalse(self.router_r1.routing_table.has_route("R3"))
        self.assertFalse(self.router_r2.routing_table.has_route("R3"))
        self.assertIsNone(self.router_r1.routing_table.get_route("R3"))

    def test_refresh_all_build_failure_aborts_without_updating_any_router(self):
        """
        Verify that if route calculation fails for any router during refresh_all():
        1. ResilienceError is raised identifying the offending router.
        2. The original exception is chained via __cause__.
        3. No router's existing routing table is replaced (all-or-nothing transactional guarantee).
        """
        mgr = TopologyRoutingManager(
            self.topo,
            {"R1": self.router_r1, "R2": self.router_r2},
        )
        old_table_r1 = self.router_r1.routing_table
        old_table_r2 = self.router_r2.routing_table

        original_build = build_routing_table

        def flaky_build(topo, node_id):
            if node_id == "R2":
                raise RuntimeError("Simulated route calculation crash for R2")
            return original_build(topo, node_id)

        with mock.patch(
            "NetworkPacketRouting.resilience.build_routing_table",
            side_effect=flaky_build,
        ):
            with self.assertRaises(ResilienceError) as ctx:
                mgr.refresh_all()

            # Error identifies offending router R2
            self.assertIn("R2", str(ctx.exception))
            # Cause is chained
            self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
            self.assertEqual(
                str(ctx.exception.__cause__),
                "Simulated route calculation crash for R2",
            )

        # Transactional guarantee: neither router was updated
        self.assertIs(self.router_r1.routing_table, old_table_r1)
        self.assertIs(self.router_r2.routing_table, old_table_r2)

    def test_disconnect_link_build_failure_preserves_live_topology_and_router_tables(self):
        """
        Verify that if route calculation fails during prospective preparation in disconnect_link():
        1. ResilienceError is raised identifying the router and original cause.
        2. The live topology is NOT altered (link remains intact).
        3. No managed router has its routing table modified.
        """
        mgr = TopologyRoutingManager(
            self.topo,
            {"R1": self.router_r1, "R2": self.router_r2},
        )
        old_table_r1 = self.router_r1.routing_table
        old_table_r2 = self.router_r2.routing_table

        self.assertTrue(self.topo.has_link("R1", "R2"))

        with mock.patch(
            "NetworkPacketRouting.resilience.build_routing_table",
            side_effect=RuntimeError("Simulated route planner crash during disconnect"),
        ):
            with self.assertRaises(ResilienceError) as ctx:
                mgr.disconnect_link("R1", "R2")

            self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
            self.assertIn("R1", str(ctx.exception))

        # Crucial check: Live topology was NOT changed!
        self.assertTrue(self.topo.has_link("R1", "R2"))
        # Routers were not modified
        self.assertIs(self.router_r1.routing_table, old_table_r1)
        self.assertIs(self.router_r2.routing_table, old_table_r2)

    def test_disconnect_link_installation_failure_documents_partial_state(self):
        """
        Verify that if candidate tables succeed but an unexpected installation error
        occurs on a router during disconnect_link(), ResilienceError is raised and
        documents the partial failure state.
        """
        mgr = TopologyRoutingManager(
            self.topo,
            {"R1": self.router_r1, "R2": self.router_r2},
        )
        self.assertTrue(self.topo.has_link("R1", "R2"))

        with mock.patch.object(
            self.router_r2,
            "update_routing_table",
            side_effect=TypeError("Simulated installation type error"),
        ):
            with self.assertRaises(ResilienceError) as ctx:
                mgr.disconnect_link("R1", "R2")

            err_msg = str(ctx.exception)
            self.assertIn("Topology link ('R1' - 'R2') was severed", err_msg)
            self.assertIn("R2", err_msg)
            self.assertIn("Simulated installation type error", err_msg)

        # The topology link was indeed removed before installation failed
        self.assertFalse(self.topo.has_link("R1", "R2"))

    def test_refresh_all_installation_failure_raises_resilience_error(self):
        """Verify that an installation failure in refresh_all() raises ResilienceError with details."""
        mgr = TopologyRoutingManager(
            self.topo,
            {"R1": self.router_r1, "R2": self.router_r2},
        )
        with mock.patch.object(
            self.router_r1,
            "update_routing_table",
            side_effect=RuntimeError("Simulated install failure for R1"),
        ):
            with self.assertRaises(ResilienceError) as ctx:
                mgr.refresh_all()

            self.assertIn("R1", str(ctx.exception))
            self.assertIn("Simulated install failure for R1", str(ctx.exception))

    def test_candidate_topology_preserves_insertion_order_tie_breaking(self):
        """
        Verify that the deep-copied candidate topology in disconnect_link() preserves
        neighbor insertion order and deterministic BFS tie-breaking.

        Topology:
            R1 connected to R_primary, then R_backup1, then R_backup2.
            All three connect to target node 'DEST'.
            Initial path R1 -> DEST: hops=2 via R_primary.
            Sever R1-R_primary.
            Candidate topology has neighbors [R_backup1, R_backup2] for R1.
            Both have hop-count 2 to DEST.
            BFS must choose R_backup1 (inserted first) rather than R_backup2.
        """
        topo = NetworkTopology()
        topo.register_node("R1", "127.0.0.1", 6001)
        topo.register_node("R_primary", "127.0.0.1", 6002)
        topo.register_node("R_backup1", "127.0.0.1", 6003)
        topo.register_node("R_backup2", "127.0.0.1", 6004)
        topo.register_node("DEST", "127.0.0.1", 6005)

        # Insertion order at R1: R_primary, R_backup1, R_backup2
        topo.connect("R1", "R_primary")
        topo.connect("R1", "R_backup1")
        topo.connect("R1", "R_backup2")

        topo.connect("R_primary", "DEST")
        topo.connect("R_backup1", "DEST")
        topo.connect("R_backup2", "DEST")

        r1 = create_topology_router(topo, "R1")
        # Initially, next hop to DEST is R_primary
        self.assertEqual(r1.routing_table.get_route("DEST"), ("127.0.0.1", 6002))

        mgr = TopologyRoutingManager(topo, {"R1": r1})
        self.assertTrue(mgr.disconnect_link("R1", "R_primary"))

        # After disconnecting primary, next hop must deterministically be R_backup1
        self.assertEqual(r1.routing_table.get_route("DEST"), ("127.0.0.1", 6003))


class TestMultiHopResilienceIntegration(unittest.TestCase):
    """
    Multi-hop dynamic rerouting integration test across real TCP sockets.

    Topology:
        Initially:
            R1 — R2 — receiver-1
             \\  /
              R3

        R1 links: to R2, to R3.
        R2 links: to R1, to R3, to receiver-1.
        R3 links: to R1, to R2.
        receiver-1 links: to R2.

        Initially, shortest path R1 -> receiver-1 is:
            R1 -> R2 -> receiver-1 (2 hops).
            R1's next hop to receiver-1 is R2.

        Sever link R1-R2:
        Alternate shortest path R1 -> receiver-1 is:
            R1 -> R3 -> R2 -> receiver-1 (3 hops).
            R1's next hop to receiver-1 becomes R3.

        Packet with initial TTL 8 arrives at receiver-1 with TTL 5
        (decremented at R1: 8->7, at R3: 7->6, at R2: 6->5).
    """

    def setUp(self):
        self.routers: List[Router] = []
        self.threads: List[threading.Thread] = []
        self.sockets: List[socket.socket] = []

    def tearDown(self):
        """Clean up all sockets, stop all routers, and terminate background threads."""
        for sock in self.sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

        for router in self.routers:
            try:
                router.stop()
            except Exception:
                pass

        for thread in self.threads:
            thread.join(timeout=2.0)

    def _start_router(self, router: Router) -> None:
        """Starts a router and serves connections in a background thread."""
        router.start()
        self.routers.append(router)
        thread = threading.Thread(target=router.serve_forever, daemon=True)
        thread.start()
        self.threads.append(thread)

    def test_dynamic_reroute_after_link_failure(self):
        """
        Verify that after link failure between R1 and R2:
        1. R1 dynamically updates its forwarding table to route through R3.
        2. Packets navigate R1 -> R3 -> R2 -> receiver-1.
        3. Initial TTL 8 arrives with TTL 5.
        4. Payload and packet ID arrive intact.
        5. Sender receives DELIVERED response from R1.
        6. Independent receiver confirms final delivery.
        """
        # 1. Allocate distinct ephemeral ports
        port_r1 = _find_free_port()
        port_r2 = _find_free_port()
        port_r3 = _find_free_port()
        while len({port_r1, port_r2, port_r3}) < 3:
            port_r2 = _find_free_port()
            port_r3 = _find_free_port()

        # Real receiver TCP listener
        recv_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        recv_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_listener.bind(("127.0.0.1", 0))
        recv_listener.listen(5)
        recv_listener.settimeout(4.0)
        self.sockets.append(recv_listener)
        port_receiver = recv_listener.getsockname()[1]

        # 2. Build initial topology
        topo = NetworkTopology()
        topo.register_node("R1", "127.0.0.1", port_r1)
        topo.register_node("R2", "127.0.0.1", port_r2)
        topo.register_node("R3", "127.0.0.1", port_r3)
        topo.register_node("receiver-1", "127.0.0.1", port_receiver)

        topo.connect("R1", "R2")
        topo.connect("R1", "R3")
        topo.connect("R3", "R2")
        topo.connect("R2", "receiver-1")

        # 3. Create real routers
        router_r1 = create_topology_router(topo, "R1", downstream_timeout=1.5)
        router_r2 = create_topology_router(topo, "R2", downstream_timeout=1.5)
        router_r3 = create_topology_router(topo, "R3", downstream_timeout=1.5)

        # Confirm initial next hop from R1 to receiver-1 is R2
        self.assertEqual(
            router_r1.routing_table.get_route("receiver-1"),
            ("127.0.0.1", port_r2),
        )

        # 4. Attach TopologyRoutingManager
        mgr = TopologyRoutingManager(
            topo,
            {"R1": router_r1, "R2": router_r2, "R3": router_r3},
        )

        # 5. Start routers
        self._start_router(router_r1)
        self._start_router(router_r2)
        self._start_router(router_r3)

        # 6. Sever the primary R1-R2 link and trigger dynamic refresh
        link_removed = mgr.disconnect_link("R1", "R2")
        self.assertTrue(link_removed)

        # Verify that R1's next hop for receiver-1 changed from R2 to R3
        self.assertEqual(
            router_r1.routing_table.get_route("receiver-1"),
            ("127.0.0.1", port_r3),
        )
        # R3 forwards to R2
        self.assertEqual(
            router_r3.routing_table.get_route("receiver-1"),
            ("127.0.0.1", port_r2),
        )
        # R2 forwards directly to receiver-1
        self.assertEqual(
            router_r2.routing_table.get_route("receiver-1"),
            ("127.0.0.1", port_receiver),
        )

        # 7. Connect client sender to R1
        sender_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sender_sock.settimeout(3.0)
        sender_sock.connect(("127.0.0.1", port_r1))
        self.sockets.append(sender_sock)

        # 8. Send packet with initial TTL = 8
        test_packet = Packet(
            version=1,
            src="sender-1",
            dest="receiver-1",
            payload="Rerouted through R3 after R1-R2 severed!",
            packet_id="pkt-resilience-42",
            ttl=8,
        )
        send_frame(sender_sock, test_packet.to_json_bytes())

        # 9. Accept packet at final receiver listener
        receiver_conn, addr = recv_listener.accept()
        self.sockets.append(receiver_conn)
        receiver_conn.settimeout(3.0)

        received_bytes = recv_frame(receiver_conn)
        self.assertIsNotNone(received_bytes, "Receiver received EOF instead of packet")
        received_packet = Packet.from_json_bytes(received_bytes)

        # 10. Verify packet received at destination
        self.assertEqual(received_packet.packet_id, "pkt-resilience-42")
        self.assertEqual(received_packet.src, "sender-1")
        self.assertEqual(received_packet.dest, "receiver-1")
        self.assertEqual(
            received_packet.payload,
            "Rerouted through R3 after R1-R2 severed!",
        )
        # R1 (8 -> 7), R3 (7 -> 6), R2 (6 -> 5)
        self.assertEqual(received_packet.ttl, 5)

        # 11. Verify sender gets correlated DELIVERED response from R1
        response_bytes = recv_frame(sender_sock)
        self.assertIsNotNone(response_bytes)
        response = RouterResponse.from_json_bytes(response_bytes)

        self.assertEqual(response.packet_id, "pkt-resilience-42")
        self.assertEqual(response.status, "DELIVERED")
        self.assertEqual(response.version, 1)

        try:
            sender_sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class TestWeightedResilienceManager(unittest.TestCase):
    """Unit tests for routing mode management in TopologyRoutingManager."""

    def setUp(self):
        self.topo = NetworkTopology()
        self.topo.register_node("R1", "127.0.0.1", 5001)
        self.topo.register_node("R2", "127.0.0.1", 5002)
        self.topo.register_node("D", "127.0.0.1", 5003)

        self.topo.connect("R1", "D", weight=100)
        self.topo.connect("R1", "R2", weight=1)
        self.topo.connect("R2", "D", weight=1)

        self.router_r1 = create_topology_router(self.topo, "R1", routing_mode="weighted")
        self.router_r2 = create_topology_router(self.topo, "R2", routing_mode="weighted")

    def test_manager_initialization_with_routing_modes(self):
        """Verify manager initializes and respects routing_modes mapping."""
        mgr = TopologyRoutingManager(
            self.topo,
            {"R1": self.router_r1, "R2": self.router_r2},
            routing_modes={"R1": "weighted", "R2": "unweighted"},
        )
        self.assertEqual(mgr.get_routing_mode("R1"), "weighted")
        self.assertEqual(mgr.get_routing_mode("R2"), "unweighted")

    def test_manager_invalid_routing_modes_raise(self):
        """Verify invalid routing modes at initialization raise InvalidRoutingModeError."""
        with self.assertRaises(InvalidRoutingModeError):
            TopologyRoutingManager(
                self.topo,
                {"R1": self.router_r1},
                routing_modes={"R1": "bogus_mode"},
            )
        with self.assertRaises(InvalidRoutingModeError):
            TopologyRoutingManager(
                self.topo,
                default_routing_mode="invalid_default",
            )
        with self.assertRaises(InvalidManagerConfigError):
            TopologyRoutingManager(
                self.topo,
                routing_modes="not-a-dict",  # type: ignore
            )

    def test_get_and_set_routing_mode(self):
        """Verify dynamic retrieval and mutation of a managed router's routing mode."""
        mgr = TopologyRoutingManager(self.topo, {"R1": self.router_r1})
        self.assertEqual(mgr.get_routing_mode("R1"), "unweighted")

        mgr.set_routing_mode("R1", "weighted")
        self.assertEqual(mgr.get_routing_mode("R1"), "weighted")

        with self.assertRaises(InvalidRoutingModeError):
            mgr.set_routing_mode("R1", "unknown")

        with self.assertRaises(RouterNotManagedError):
            mgr.get_routing_mode("ghost")

        with self.assertRaises(RouterNotManagedError):
            mgr.set_routing_mode("ghost", "weighted")

    def test_refresh_router_uses_configured_mode(self):
        """Verify refresh_router recalculates table according to router's routing mode."""
        mgr = TopologyRoutingManager(
            self.topo,
            {"R1": self.router_r1},
            routing_modes={"R1": "weighted"},
        )
        # Weighted mode: R1 -> D chooses R2 (cost 2 vs 100)
        table_weighted = mgr.refresh_router("R1")
        self.assertEqual(table_weighted.get_route("D"), ("127.0.0.1", 5002))

        # Switch to unweighted mode
        mgr.set_routing_mode("R1", "unweighted")
        table_unweighted = mgr.refresh_router("R1")
        # Unweighted BFS mode: R1 -> D chooses direct link (1 hop)
        self.assertEqual(table_unweighted.get_route("D"), ("127.0.0.1", 5003))

    def test_refresh_all_preserves_per_router_modes(self):
        """Verify refresh_all computes each router's table with its specific mode."""
        mgr = TopologyRoutingManager(
            self.topo,
            {"R1": self.router_r1, "R2": self.router_r2},
            routing_modes={"R1": "weighted", "R2": "unweighted"},
        )
        tables = mgr.refresh_all()
        # R1 is weighted: routes to D via R2 (cost 2)
        self.assertEqual(tables["R1"].get_route("D"), ("127.0.0.1", 5002))
        # R2 is unweighted: routes to D directly (1 hop)
        self.assertEqual(tables["R2"].get_route("D"), ("127.0.0.1", 5003))

    def test_disconnect_link_preserves_weighted_mode(self):
        """
        Verify disconnect_link prepares candidate tables using each router's configured
        routing mode, and that weighted routing is preserved after the refresh.

        Topology (setUp):
            R1 --[100]-- D (127.0.0.1:5003)
            R1 --[1]--  R2 (127.0.0.1:5002)
            R2 --[1]--  D

        After severing R1-R2:
          - R1's cheapest path to D is now via the direct link (cost 100).
            next hop for D = D's endpoint (127.0.0.1:5003).
          - R2 remains reachable from R1 only through D (R1->D->R2, cost 101).
            next hop for R2 = D's endpoint (127.0.0.1:5003).
          - The manager's routing mode for R1 must still be 'weighted'
            (disconnect_link must not reset or change routing modes).
        """
        mgr = TopologyRoutingManager(
            self.topo,
            {"R1": self.router_r1, "R2": self.router_r2},
            routing_modes={"R1": "weighted", "R2": "weighted"},
        )
        self.assertTrue(mgr.disconnect_link("R1", "R2"))

        # 1. R1's route to D uses the direct link (cheapest remaining path, cost 100).
        #    next hop = D's registered endpoint.
        d_endpoint = self.topo.get_endpoint("D")
        self.assertEqual(
            self.router_r1.routing_table.get_route("D"),
            d_endpoint,
        )

        # 2. R2 is still reachable via R1->D->R2 (cost 101).  The Dijkstra path goes
        #    through D first, so the next hop for R2 must also be D's endpoint.
        self.assertTrue(
            self.router_r1.routing_table.has_route("R2"),
            "R2 must still be reachable from R1 via the indirect path R1->D->R2",
        )
        self.assertEqual(
            self.router_r1.routing_table.get_route("R2"),
            d_endpoint,
            "Next hop to R2 must be D (intermediate on the only remaining path)",
        )

        # 3. Routing mode is preserved — disconnect_link must not reset the mode.
        self.assertEqual(
            mgr.get_routing_mode("R1"),
            "weighted",
            "disconnect_link must not alter the routing mode of managed routers",
        )

    def test_connect_link_with_weight_and_two_phase_transaction(self):
        """Verify connect_link accepts optional weight and updates weighted routers."""
        topo = NetworkTopology()
        topo.register_node("A", "127.0.0.1", 6001)
        topo.register_node("B", "127.0.0.1", 6002)
        topo.register_node("C", "127.0.0.1", 6003)

        topo.connect("A", "C", weight=50)

        router_a = create_topology_router(topo, "A", routing_mode="weighted")
        mgr = TopologyRoutingManager(
            topo,
            {"A": router_a},
            routing_modes={"A": "weighted"},
        )
        self.assertEqual(router_a.routing_table.get_route("C"), ("127.0.0.1", 6003))

        # Add link A - B with weight 1, and B - C with weight 1
        self.assertTrue(mgr.connect_link("A", "B", weight=1))
        self.assertTrue(mgr.connect_link("B", "C", weight=1))

        # Now A -> C routes via B (cost 2 vs 50)
        self.assertEqual(router_a.routing_table.get_route("C"), ("127.0.0.1", 6002))
        self.assertEqual(topo.get_link_weight("A", "B"), 1)
        self.assertEqual(topo.get_link_weight("B", "C"), 1)

    def test_connect_link_candidate_failure_preserves_live_topology(self):
        """Verify that if candidate route preparation fails in connect_link, topology is not changed."""
        mgr = TopologyRoutingManager(
            self.topo,
            {"R1": self.router_r1},
            routing_modes={"R1": "weighted"},
        )
        self.topo.register_node("NewNode", "127.0.0.1", 5099)
        self.assertFalse(self.topo.has_link("R1", "NewNode"))

        with mock.patch(
            "NetworkPacketRouting.resilience.build_weighted_routing_table",
            side_effect=RuntimeError("Simulated route crash during connect_link"),
        ):
            with self.assertRaises(ResilienceError) as ctx:
                mgr.connect_link("R1", "NewNode", weight=5)

            self.assertIn("R1", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, RuntimeError)

        # Topology link was NOT added to live topology
        self.assertFalse(self.topo.has_link("R1", "NewNode"))


class TestMultiHopWeightedResilienceIntegration(unittest.TestCase):
    """Real TCP socket multi-hop integration tests for weighted routing and zero-weight loop safety."""

    def setUp(self):
        self.routers: List[Router] = []
        self.threads: List[threading.Thread] = []
        self.sockets: List[socket.socket] = []

    def tearDown(self):
        for sock in self.sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

        for router in self.routers:
            try:
                router.stop()
            except Exception:
                pass

        for thread in self.threads:
            thread.join(timeout=2.0)

    def _start_router(self, router: Router) -> None:
        router.start()
        self.routers.append(router)
        thread = threading.Thread(target=router.serve_forever, daemon=True)
        thread.start()
        self.threads.append(thread)

    def test_multi_hop_tcp_weighted_forwarding_zero_weight_links_no_loop(self):
        """
        Verify that multiple routers with zero-weight links deliver packets without looping.

        Topology:
            Sender -> R1 ---[cost 0]--- R2 ---[cost 1]--- receiver-1
                      R1 ---[cost 10]--- receiver-1

        R1 and R2 are managed routers with routing_mode='weighted'.
        Packet with initial TTL=8 sent to R1 destined for receiver-1:
        - R1 forwards to R2 across the zero-weight link (R1 TTL 8 -> 7).
        - R2 forwards to receiver-1 (R2 TTL 7 -> 6).
        - Packet arrives at receiver-1 with TTL=6.
        - Sender receives DELIVERED from R1.
        - Zero loops or livelocks occur.
        """
        port_r1 = _find_free_port()
        port_r2 = _find_free_port()
        while port_r2 == port_r1:
            port_r2 = _find_free_port()

        recv_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        recv_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_listener.bind(("127.0.0.1", 0))
        recv_listener.listen(5)
        recv_listener.settimeout(3.0)
        self.sockets.append(recv_listener)
        port_receiver = recv_listener.getsockname()[1]

        topo = NetworkTopology()
        topo.register_node("R1", "127.0.0.1", port_r1)
        topo.register_node("R2", "127.0.0.1", port_r2)
        topo.register_node("receiver-1", "127.0.0.1", port_receiver)

        topo.connect("R1", "receiver-1", weight=10)
        topo.connect("R1", "R2", weight=0)
        topo.connect("R2", "receiver-1", weight=1)

        router_r1 = create_topology_router(
            topo, "R1", downstream_timeout=1.5, routing_mode="weighted"
        )
        router_r2 = create_topology_router(
            topo, "R2", downstream_timeout=1.5, routing_mode="weighted"
        )

        mgr = TopologyRoutingManager(
            topo,
            {"R1": router_r1, "R2": router_r2},
            routing_modes={"R1": "weighted", "R2": "weighted"},
        )

        self._start_router(router_r1)
        self._start_router(router_r2)

        sender_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sender_sock.settimeout(3.0)
        self.sockets.append(sender_sock)
        sender_sock.connect(("127.0.0.1", port_r1))

        pkt = Packet(
            version=1,
            src="sender-zero-wt",
            dest="receiver-1",
            payload="Zero-weight link loop-freedom test",
            packet_id="pkt-zero-wt-1",
            ttl=8,
        )
        send_frame(sender_sock, pkt.to_json_bytes())

        # Sender gets DELIVERED from R1
        resp_bytes = recv_frame(sender_sock)
        self.assertIsNotNone(resp_bytes)
        resp = RouterResponse.from_json_bytes(resp_bytes)
        self.assertEqual(resp.status, "DELIVERED")

        # Receiver accepts and verifies packet
        conn, _ = recv_listener.accept()
        self.sockets.append(conn)
        conn.settimeout(3.0)
        rx_bytes = recv_frame(conn)
        self.assertIsNotNone(rx_bytes)
        rx_pkt = Packet.from_json_bytes(rx_bytes)
        self.assertEqual(rx_pkt.packet_id, "pkt-zero-wt-1")
        self.assertEqual(rx_pkt.payload, "Zero-weight link loop-freedom test")
        self.assertEqual(rx_pkt.ttl, 6)

        try:
            sender_sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        try:
            conn.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    def test_multi_hop_tcp_resilience_rerouting_weighted(self):
        """
        Verify live TCP rerouting when a link is disconnected in weighted mode.

        1. Initial: R1 -> R2 (weight 1) -> receiver-1 (weight 1) [cost 2, 2 hops]
           Alternate: R1 -> receiver-1 (weight 100) [cost 100, 1 hop]
        2. First packet sent: traverses R1 -> R2 -> receiver-1 (TTL 8 -> 6).
        3. Sever R1 - R2 via disconnect_link().
        4. R1 updates forwarding table to direct link (cost 100).
        5. Second packet sent: traverses R1 -> receiver-1 directly (TTL 8 -> 7).
        """
        port_r1 = _find_free_port()
        port_r2 = _find_free_port()
        while port_r2 == port_r1:
            port_r2 = _find_free_port()

        recv_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        recv_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_listener.bind(("127.0.0.1", 0))
        recv_listener.listen(5)
        recv_listener.settimeout(3.0)
        self.sockets.append(recv_listener)
        port_receiver = recv_listener.getsockname()[1]

        topo = NetworkTopology()
        topo.register_node("R1", "127.0.0.1", port_r1)
        topo.register_node("R2", "127.0.0.1", port_r2)
        topo.register_node("receiver-1", "127.0.0.1", port_receiver)

        topo.connect("R1", "receiver-1", weight=100)
        topo.connect("R1", "R2", weight=1)
        topo.connect("R2", "receiver-1", weight=1)

        router_r1 = create_topology_router(
            topo, "R1", downstream_timeout=1.5, routing_mode="weighted"
        )
        router_r2 = create_topology_router(
            topo, "R2", downstream_timeout=1.5, routing_mode="weighted"
        )

        mgr = TopologyRoutingManager(
            topo,
            {"R1": router_r1, "R2": router_r2},
            routing_modes={"R1": "weighted", "R2": "weighted"},
        )

        self._start_router(router_r1)
        self._start_router(router_r2)

        # Transmission 1: R1 -> R2 -> receiver-1
        sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock1.settimeout(3.0)
        self.sockets.append(sock1)
        sock1.connect(("127.0.0.1", port_r1))

        pkt1 = Packet(
            version=1,
            src="sender-1",
            dest="receiver-1",
            payload="Before disconnect",
            packet_id="pkt-before-disconnect",
            ttl=8,
        )
        send_frame(sock1, pkt1.to_json_bytes())
        resp1 = RouterResponse.from_json_bytes(recv_frame(sock1))
        self.assertEqual(resp1.status, "DELIVERED")

        conn1, _ = recv_listener.accept()
        self.sockets.append(conn1)
        conn1.settimeout(3.0)
        rx1 = Packet.from_json_bytes(recv_frame(conn1))
        self.assertEqual(rx1.ttl, 6)

        # Half-close sock1 so R1's synchronous handler sees clean EOF and exits
        # the client loop, freeing R1 to accept the next connection (sock2).
        # The router is single-threaded; without this it would block waiting for
        # the next frame from sock1 and never call accept() again.
        try:
            sock1.shutdown(socket.SHUT_WR)
        except OSError:
            pass

        # Sever link R1 - R2
        self.assertTrue(mgr.disconnect_link("R1", "R2"))

        # Transmission 2: R1 -> receiver-1 directly
        sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock2.settimeout(3.0)
        self.sockets.append(sock2)
        sock2.connect(("127.0.0.1", port_r1))

        pkt2 = Packet(
            version=1,
            src="sender-1",
            dest="receiver-1",
            payload="After disconnect",
            packet_id="pkt-after-disconnect",
            ttl=8,
        )
        send_frame(sock2, pkt2.to_json_bytes())
        resp2 = RouterResponse.from_json_bytes(recv_frame(sock2))
        self.assertEqual(resp2.status, "DELIVERED")

        conn2, _ = recv_listener.accept()
        self.sockets.append(conn2)
        conn2.settimeout(3.0)
        rx2 = Packet.from_json_bytes(recv_frame(conn2))
        self.assertEqual(rx2.ttl, 7)  # Decremented only once at R1

        # Half-close sock2 symmetrically so R1's handler exits cleanly.
        try:
            sock2.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    def test_zero_weight_cycle_three_routers_no_loop(self):
        """
        Verify that independently generated routing tables for three routers
        forming a zero-weight triangle do not produce a forwarding loop.

        Topology:
            R1 --[0]-- R2 --[0]-- R3 --[0]-- R1   (triangle, all zero-weight)
            R2 --[1]-- receiver

        Edge insertion order at each node (controls Dijkstra tie-breaking):
            R1 neighbors: R2 (inserted first), R3
            R2 neighbors: R1, R3, receiver
            R3 neighbors: R2 (inserted first), R1

        Every router-pair has two equal-cost paths through the zero-weight
        triangle (cost 0 direct, cost 0 via third router). Cost to receiver
        is 1 from R2 directly, and 1 from R1 or R3 via R2.

        Expected Dijkstra results (weighted, strict-improvement tie-breaking):
            R1 → receiver: next hop = R2 (R1→R2 cost 0, then R2→receiver cost 1)
            R2 → receiver: next hop = receiver (direct, cost 1)
            R3 → receiver: next hop = R2 (R3→R2 cost 0, then R2→receiver cost 1)
              -- R3 may also route via R1 (equal cost), but either produces correct delivery

        Loop impossibility argument:
            With non-negative weights, Dijkstra's predecessor tree is loop-free by
            construction (each node is settled exactly once via the min-cost path).
            A forwarding loop between two routers Ra and Rb toward destination D
            would require cost(Ra→D) > cost(Ra→Rb→D) AND cost(Rb→D) > cost(Rb→Ra→D),
            which gives cost(Ra→D) > cost(Ra→Rb) + cost(Rb→Ra) + cost(Ra→D),
            requiring a negative cycle — impossible with non-negative weights.

        The test exercises the independently generated routing tables of THREE
        routers, not just the Dijkstra internals. TTL is the independent witness:
            - A correctly non-looping packet R1→R2→receiver decrements TTL twice (8→6).
            - Any loop would consume extra TTL and either arrive with lower TTL or
              hit DROPPED_TTL before reaching receiver.
        A second packet entering via R3 further exercises equal-cost tie-breaking.
        """
        # -- allocate ports -------------------------------------------------
        port_r1 = _find_free_port()
        port_r2 = _find_free_port()
        port_r3 = _find_free_port()
        while len({port_r1, port_r2, port_r3}) < 3:
            port_r2 = _find_free_port()
            port_r3 = _find_free_port()

        recv_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        recv_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_listener.bind(("127.0.0.1", 0))
        recv_listener.listen(5)
        recv_listener.settimeout(3.0)
        self.sockets.append(recv_listener)
        port_receiver = recv_listener.getsockname()[1]

        # -- build topology -------------------------------------------------
        topo = NetworkTopology()
        topo.register_node("R1", "127.0.0.1", port_r1)
        topo.register_node("R2", "127.0.0.1", port_r2)
        topo.register_node("R3", "127.0.0.1", port_r3)
        topo.register_node("receiver-1", "127.0.0.1", port_receiver)

        # Zero-weight triangle: insertion order determines Dijkstra tie-breaking.
        # R1: connects to R2 first, then R3.
        # R3: connects to R2 (via R2-R3 link above) first by adjacency, then R1.
        topo.connect("R1", "R2", weight=0)   # R1 adj: [R2]; R2 adj: [R1]
        topo.connect("R2", "R3", weight=0)   # R2 adj: [R1, R3]; R3 adj: [R2]
        topo.connect("R3", "R1", weight=0)   # R3 adj: [R2, R1]; R1 adj: [R2, R3]
        topo.connect("R2", "receiver-1", weight=1)  # R2 adj: [R1, R3, receiver-1]

        # -- create weighted routers ----------------------------------------
        router_r1 = create_topology_router(
            topo, "R1", downstream_timeout=1.5, routing_mode="weighted"
        )
        router_r2 = create_topology_router(
            topo, "R2", downstream_timeout=1.5, routing_mode="weighted"
        )
        router_r3 = create_topology_router(
            topo, "R3", downstream_timeout=1.5, routing_mode="weighted"
        )

        # -- assert routing tables before starting TCP ----------------------
        # R1: receiver-1 must route via R2 (first zero-weight neighbor, cost 0+1=1)
        self.assertEqual(
            router_r1.routing_table.get_route("receiver-1"),
            ("127.0.0.1", port_r2),
            "R1 must forward to receiver-1 via R2 (first zero-weight neighbor)",
        )
        # R2: receiver-1 is a direct neighbor (cost 1)
        self.assertEqual(
            router_r2.routing_table.get_route("receiver-1"),
            ("127.0.0.1", port_receiver),
            "R2 must forward to receiver-1 directly",
        )
        # R3: receiver-1 must route via one of the zero-weight neighbors (R2 or R1).
        # Both give cost 1 to receiver-1; the actual next hop is implementation-defined
        # by tie-breaking. What must NOT happen: R3→R1 and R1→R3 (mutual loop).
        r3_next_hop = router_r3.routing_table.get_route("receiver-1")
        self.assertIsNotNone(r3_next_hop, "R3 must have a route to receiver-1")
        r3_next_hop_port = r3_next_hop[1]
        # R3's next hop must be either R2 directly or R1 (who itself points to R2).
        # Both are valid non-looping choices.  The loop scenario (R3→R1, R1→R3)
        # is ruled out because R1's next hop for receiver-1 is R2, not R3.
        self.assertIn(
            r3_next_hop_port,
            {port_r2, port_r1},
            "R3's next hop to receiver-1 must be either R2 or R1 (not R3 itself)",
        )
        # Critical anti-loop invariant: confirm R1 does NOT route to R3 for receiver-1.
        # If R1→R3 and R3→R1 simultaneously, that is a forwarding loop.
        self.assertEqual(
            router_r1.routing_table.get_route("receiver-1"),
            ("127.0.0.1", port_r2),
            "R1 routing to R3 while R3 routes to R1 would be a forwarding loop",
        )

        # -- start routers --------------------------------------------------
        self._start_router(router_r1)
        self._start_router(router_r2)
        self._start_router(router_r3)

        # -- Transmission 1: sender → R1 → R2 → receiver -------------------
        # Expected TTL decrements: R1 (8→7), R2 (7→6). Arrives with TTL=6.
        sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock1.settimeout(3.0)
        self.sockets.append(sock1)
        sock1.connect(("127.0.0.1", port_r1))

        pkt1 = Packet(
            version=1,
            src="sender-zero-cycle",
            dest="receiver-1",
            payload="Zero-weight-cycle loop-free test via R1",
            packet_id="pkt-zero-cycle-r1",
            ttl=8,
        )
        send_frame(sock1, pkt1.to_json_bytes())
        resp1 = RouterResponse.from_json_bytes(recv_frame(sock1))
        self.assertEqual(resp1.status, "DELIVERED",
                         "R1 must deliver pkt to receiver-1 without looping")
        self.assertEqual(resp1.packet_id, "pkt-zero-cycle-r1")

        conn1, _ = recv_listener.accept()
        self.sockets.append(conn1)
        conn1.settimeout(3.0)
        rx1 = Packet.from_json_bytes(recv_frame(conn1))
        self.assertEqual(rx1.payload, "Zero-weight-cycle loop-free test via R1")
        # TTL=8 → 7 at R1 → 6 at R2. Any loop via R3 would further decrement.
        self.assertEqual(
            rx1.ttl, 6,
            "TTL must be exactly 6 (two hops: R1 and R2). Any extra decrement "
            "indicates a forwarding cycle through the zero-weight triangle.",
        )

        # Half-close sock1 so R1's synchronous handler exits and R1 can accept sock3.
        try:
            sock1.shutdown(socket.SHUT_WR)
        except OSError:
            pass

        # -- Transmission 2: sender → R3 → receiver (via R3's chosen next hop) --
        # R3's Dijkstra picks either R2 or R1; both cost-1 paths reach receiver.
        # If R3→R1: path is R3→R1→R2→receiver (TTL: 8→7→6→5, arrives TTL=5).
        # If R3→R2: path is R3→R2→receiver (TTL: 8→7→6, arrives TTL=6).
        # Either is correct forwarding. What's NOT allowed: looping (TTL < 5).
        sock3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock3.settimeout(3.0)
        self.sockets.append(sock3)
        sock3.connect(("127.0.0.1", port_r3))

        pkt3 = Packet(
            version=1,
            src="sender-zero-cycle",
            dest="receiver-1",
            payload="Zero-weight-cycle loop-free test via R3",
            packet_id="pkt-zero-cycle-r3",
            ttl=8,
        )
        send_frame(sock3, pkt3.to_json_bytes())
        resp3 = RouterResponse.from_json_bytes(recv_frame(sock3))
        self.assertEqual(resp3.status, "DELIVERED",
                         "R3 must deliver pkt to receiver-1 without looping")
        self.assertEqual(resp3.packet_id, "pkt-zero-cycle-r3")

        conn3, _ = recv_listener.accept()
        self.sockets.append(conn3)
        conn3.settimeout(3.0)
        rx3 = Packet.from_json_bytes(recv_frame(conn3))
        self.assertEqual(rx3.payload, "Zero-weight-cycle loop-free test via R3")
        # R3→R2→receiver: TTL 8→7→6 (2 decrements). Or R3→R1→R2→receiver: TTL 8→7→6→5 (3 decrements).
        # A loop would give TTL ≤ 4.  Assert a minimum bound that proves no loop.
        self.assertIn(
            rx3.ttl, {5, 6},
            f"TTL must be 5 (3-hop path via R1) or 6 (2-hop path via R2), got {rx3.ttl}. "
            "A lower TTL would indicate an extra forwarding cycle.",
        )

        # Half-close sock3 for symmetric clean teardown.
        try:
            sock3.shutdown(socket.SHUT_WR)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
