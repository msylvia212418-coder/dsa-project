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


if __name__ == "__main__":
    unittest.main()
