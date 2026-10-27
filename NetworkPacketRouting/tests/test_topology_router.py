"""
test_topology_router.py - Unit and multi-hop integration tests for topology_router.

Verifies:
- create_topology_router() factory produces Router instances configured
  with the topology's endpoints and routing table.
- Factory rejects invalid topology types, unregistered nodes, and invalid node IDs.
- Multi-hop forwarding across real TCP sockets:
  Topology: R1 — R2 — receiver-1
  - R1 forwards to R2; R2 forwards to receiver-1.
  - TTL is decremented at each hop (e.g. 8 -> 7 -> 6).
  - Packet ID, payload, source, and destination are preserved intact.
  - Sender receives a correlated DELIVERED response from R1.
  - End-to-end receipt is independently verified at the real receiver.
  - Clean socket and thread disposal in all circumstances.
"""

from __future__ import annotations

import socket
import threading
import unittest
from typing import List, Optional, Tuple

from NetworkPacketRouting.packet import Packet, RouterResponse
from NetworkPacketRouting.protocol import recv_frame, send_frame
from NetworkPacketRouting.router import Router
from NetworkPacketRouting.topology import (
    InvalidNodeError,
    NetworkTopology,
    NodeNotFoundError,
)
from NetworkPacketRouting.topology_router import (
    InvalidRoutingModeError,
    TopologyRouterError,
    create_topology_router,
)


def _find_free_port() -> int:
    """Helper to allocate an available OS ephemeral port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestTopologyRouterFactory(unittest.TestCase):
    """Unit tests for create_topology_router factory function."""

    def setUp(self):
        self.topo = NetworkTopology()
        self.topo.register_node("R1", "127.0.0.1", 5001)
        self.topo.register_node("R2", "127.0.0.1", 5002)
        self.topo.connect("R1", "R2")

    def test_factory_creates_configured_router(self):
        """Verify the factory configures the router with the node's endpoint and routing table."""
        router = create_topology_router(self.topo, "R1")

        self.assertIsInstance(router, Router)
        self.assertEqual(router._host, "127.0.0.1")
        self.assertEqual(router._port, 5001)
        self.assertTrue(router._routing_table.has_route("R2"))
        self.assertEqual(
            router._routing_table.get_route("R2"), ("127.0.0.1", 5002)
        )
        self.assertFalse(router.is_running)

    def test_factory_defaults_to_unweighted_mode(self):
        """Verify create_topology_router defaults to unweighted BFS route generation."""
        topo = NetworkTopology()
        topo.register_node("S", "127.0.0.1", 5001)
        topo.register_node("R", "127.0.0.1", 5002)
        topo.register_node("D", "127.0.0.1", 5003)

        topo.connect("S", "D", weight=100)
        topo.connect("S", "R", weight=1)
        topo.connect("R", "D", weight=1)

        router_default = create_topology_router(topo, "S")
        # In unweighted BFS, direct 1-hop link S-D is chosen
        self.assertEqual(
            router_default.routing_table.get_route("D"),
            ("127.0.0.1", 5003),
        )

    def test_factory_explicit_weighted_mode(self):
        """Verify create_topology_router with routing_mode='weighted' uses Dijkstra."""
        topo = NetworkTopology()
        topo.register_node("S", "127.0.0.1", 5001)
        topo.register_node("R", "127.0.0.1", 5002)
        topo.register_node("D", "127.0.0.1", 5003)

        topo.connect("S", "D", weight=100)
        topo.connect("S", "R", weight=1)
        topo.connect("R", "D", weight=1)

        router_weighted = create_topology_router(topo, "S", routing_mode="weighted")
        # In weighted mode, cheaper path S -> R -> D (cost 2) is chosen over direct (cost 100)
        self.assertEqual(
            router_weighted.routing_table.get_route("D"),
            ("127.0.0.1", 5002),
        )

    def test_factory_invalid_routing_mode_raises(self):
        """Verify unsupported routing_mode raises InvalidRoutingModeError."""
        with self.assertRaises(InvalidRoutingModeError):
            create_topology_router(self.topo, "R1", routing_mode="unsupported_mode")
        with self.assertRaises(InvalidRoutingModeError):
            create_topology_router(self.topo, "R1", routing_mode="")
        with self.assertRaises(InvalidRoutingModeError):
            create_topology_router(self.topo, "R1", routing_mode="WEIGHTED")

    def test_invalid_topology_object_raises(self):
        """Verify passing a non-NetworkTopology object raises TopologyRouterError."""
        with self.assertRaises(TopologyRouterError):
            create_topology_router("not-a-topology", "R1")  # type: ignore
        with self.assertRaises(TopologyRouterError):
            create_topology_router(None, "R1")  # type: ignore
        with self.assertRaises(TopologyRouterError):
            create_topology_router({}, "R1")  # type: ignore

    def test_unregistered_node_raises_node_not_found(self):
        """Verify requesting a router for an unregistered node raises NodeNotFoundError."""
        with self.assertRaises(NodeNotFoundError):
            create_topology_router(self.topo, "ghost-router")

    def test_invalid_node_identifier_raises_invalid_node_error(self):
        """Verify invalid node identifiers raise InvalidNodeError."""
        invalid_ids = [None, 123, True, False, "", "   ", " R1", "R1 "]
        for bad_id in invalid_ids:
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(InvalidNodeError):
                    create_topology_router(self.topo, bad_id)  # type: ignore


class TestMultiHopTopologyRouting(unittest.TestCase):
    """
    End-to-end multi-hop integration tests using real TCP sockets.

    Topology under test:
        R1 — R2 — receiver-1
    """

    def setUp(self):
        self.routers: List[Router] = []
        self.threads: List[threading.Thread] = []
        self.sockets: List[socket.socket] = []

    def tearDown(self):
        """Clean up all sockets, stop all routers, and join worker threads."""
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
        """Starts a router and launches its serve_forever loop in a daemon thread."""
        router.start()
        self.routers.append(router)
        thread = threading.Thread(target=router.serve_forever, daemon=True)
        thread.start()
        self.threads.append(thread)

    def test_multi_hop_forwarding_and_ttl_decrement(self):
        """
        Verify multi-hop packet forwarding along:
            Sender -> R1 -> R2 -> receiver-1

        Checklist:
        1. Factory configures R1 and R2 with topology endpoints.
        2. R1 routes to receiver-1 via R2's endpoint.
        3. R2 routes to receiver-1 via receiver-1's endpoint.
        4. Sender transmits packet with TTL=8 to R1.
        5. Packet arrives at receiver-1 with TTL=6 (decremented twice).
        6. Sender receives DELIVERED response from R1 correlated to packet ID.
        7. Packet fields (ID, payload, src, dest) match exactly.
        """
        # 1. Allocate distinct ephemeral ports for all 3 nodes
        port_r1 = _find_free_port()
        port_r2 = _find_free_port()
        while port_r2 == port_r1:
            port_r2 = _find_free_port()

        # Set up real receiver TCP listener on an ephemeral port
        recv_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        recv_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_listener.bind(("127.0.0.1", 0))
        recv_listener.listen(5)
        recv_listener.settimeout(3.0)
        self.sockets.append(recv_listener)
        port_receiver = recv_listener.getsockname()[1]

        # 2. Build the topology: R1 - R2 - receiver-1
        topo = NetworkTopology()
        topo.register_node("R1", "127.0.0.1", port_r1)
        topo.register_node("R2", "127.0.0.1", port_r2)
        topo.register_node("receiver-1", "127.0.0.1", port_receiver)

        topo.connect("R1", "R2")
        topo.connect("R2", "receiver-1")

        # 3. Create routers via the factory
        router_r1 = create_topology_router(topo, "R1", downstream_timeout=1.5)
        router_r2 = create_topology_router(topo, "R2", downstream_timeout=1.5)

        # Verify next-hop routing entries
        # R1's next hop for receiver-1 MUST be R2's endpoint (port_r2)
        self.assertEqual(
            router_r1._routing_table.get_route("receiver-1"),
            ("127.0.0.1", port_r2),
        )
        # R2's next hop for receiver-1 MUST be receiver-1's endpoint (port_receiver)
        self.assertEqual(
            router_r2._routing_table.get_route("receiver-1"),
            ("127.0.0.1", port_receiver),
        )

        # 4. Start R1 and R2
        self._start_router(router_r1)
        self._start_router(router_r2)

        # 5. Connect sender client to R1
        sender_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sender_sock.settimeout(3.0)
        sender_sock.connect(("127.0.0.1", port_r1))
        self.sockets.append(sender_sock)

        # 6. Send packet with TTL=8
        test_packet = Packet(
            version=1,
            src="sender-1",
            dest="receiver-1",
            payload="Multi-hop integration test payload",
            packet_id="pkt-multi-hop-42",
            ttl=8,
        )
        send_frame(sender_sock, test_packet.to_json_bytes())

        # 7. Accept packet at receiver-1 listener (from R2)
        receiver_conn, addr = recv_listener.accept()
        self.sockets.append(receiver_conn)
        receiver_conn.settimeout(3.0)

        received_bytes = recv_frame(receiver_conn)
        self.assertIsNotNone(received_bytes, "Receiver received EOF instead of packet frame")
        received_packet = Packet.from_json_bytes(received_bytes)

        # 8. Verify the received packet at receiver-1
        self.assertEqual(received_packet.packet_id, "pkt-multi-hop-42")
        self.assertEqual(received_packet.src, "sender-1")
        self.assertEqual(received_packet.dest, "receiver-1")
        self.assertEqual(received_packet.payload, "Multi-hop integration test payload")
        # Decremented at R1 (8 -> 7) and decremented at R2 (7 -> 6)
        self.assertEqual(received_packet.ttl, 6)

        # 9. Verify sender receives correlated DELIVERED response from R1
        response_bytes = recv_frame(sender_sock)
        self.assertIsNotNone(response_bytes, "Sender received EOF instead of RouterResponse")
        response = RouterResponse.from_json_bytes(response_bytes)

        self.assertEqual(response.packet_id, "pkt-multi-hop-42")
        self.assertEqual(response.status, "DELIVERED")
        self.assertEqual(response.version, 1)

        # Graceful client disconnect
        try:
            sender_sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    def test_multi_hop_ttl_expiry_at_intermediate_router(self):
        """
        Verify that if TTL expires at intermediate router R2 (TTL=2 from sender):
        - R1 decrements TTL to 1 and forwards to R2.
        - R2 sees TTL=1, drops packet with DROPPED_TTL, and does not forward to receiver-1.
        - receiver-1 receives nothing.
        """
        port_r1 = _find_free_port()
        port_r2 = _find_free_port()
        while port_r2 == port_r1:
            port_r2 = _find_free_port()

        recv_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        recv_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_listener.bind(("127.0.0.1", 0))
        recv_listener.listen(5)
        recv_listener.settimeout(0.5)
        self.sockets.append(recv_listener)
        port_receiver = recv_listener.getsockname()[1]

        topo = NetworkTopology()
        topo.register_node("R1", "127.0.0.1", port_r1)
        topo.register_node("R2", "127.0.0.1", port_r2)
        topo.register_node("receiver-1", "127.0.0.1", port_receiver)
        topo.connect("R1", "R2")
        topo.connect("R2", "receiver-1")

        router_r1 = create_topology_router(topo, "R1", downstream_timeout=1.0)
        router_r2 = create_topology_router(topo, "R2", downstream_timeout=1.0)

        self._start_router(router_r1)
        self._start_router(router_r2)

        sender_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sender_sock.settimeout(2.0)
        sender_sock.connect(("127.0.0.1", port_r1))
        self.sockets.append(sender_sock)

        # Send packet with TTL=2 (R1 forwards with TTL=1, R2 must drop)
        pkt = Packet(
            version=1,
            src="sender-1",
            dest="receiver-1",
            payload="TTL drop test",
            packet_id="pkt-ttl-drop",
            ttl=2,
        )
        send_frame(sender_sock, pkt.to_json_bytes())

        # Sender gets DELIVERED from R1 (R1 successfully handed to R2)
        resp_bytes = recv_frame(sender_sock)
        self.assertIsNotNone(resp_bytes)
        resp = RouterResponse.from_json_bytes(resp_bytes)
        self.assertEqual(resp.status, "DELIVERED")

        # Receiver must NOT receive any connection or packet (timed out)
        with self.assertRaises(socket.timeout):
            recv_listener.accept()

        try:
            sender_sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    def test_multi_hop_tcp_weighted_path_cheaper_than_fewest_hop(self):
        """
        Verify real TCP forwarding chooses the lower-cost 2-hop route over a 1-hop expensive route.

        Topology:
            Sender -> R1 ---[cost 100]--- receiver-1 (1 hop, high cost)
                      R1 ---[cost 1]--- R2 ---[cost 1]--- receiver-1 (2 hops, low cost 2)

        With routing_mode='weighted':
        - R1 routes to receiver-1 via R2.
        - Packet with initial TTL=8 arrives at receiver-1 with TTL=6 (2 decrements).
        - Sender receives DELIVERED confirmation from R1.
        """
        port_r1 = _find_free_port()
        port_r2 = _find_free_port()
        while port_r2 == port_r1:
            port_r2 = _find_free_port()

        # Set up real receiver listener
        recv_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        recv_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_listener.bind(("127.0.0.1", 0))
        recv_listener.listen(5)
        recv_listener.settimeout(3.0)
        self.sockets.append(recv_listener)
        port_receiver = recv_listener.getsockname()[1]

        # Build topology
        topo = NetworkTopology()
        topo.register_node("R1", "127.0.0.1", port_r1)
        topo.register_node("R2", "127.0.0.1", port_r2)
        topo.register_node("receiver-1", "127.0.0.1", port_receiver)

        topo.connect("R1", "receiver-1", weight=100)
        topo.connect("R1", "R2", weight=1)
        topo.connect("R2", "receiver-1", weight=1)

        # Create weighted routers
        router_r1 = create_topology_router(
            topo, "R1", downstream_timeout=1.5, routing_mode="weighted"
        )
        router_r2 = create_topology_router(
            topo, "R2", downstream_timeout=1.5, routing_mode="weighted"
        )

        # Verify next hops
        self.assertEqual(
            router_r1.routing_table.get_route("receiver-1"),
            ("127.0.0.1", port_r2),
        )
        self.assertEqual(
            router_r2.routing_table.get_route("receiver-1"),
            ("127.0.0.1", port_receiver),
        )

        # Start routers
        self._start_router(router_r1)
        self._start_router(router_r2)

        # Connect sender to R1
        sender_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sender_sock.settimeout(3.0)
        self.sockets.append(sender_sock)
        sender_sock.connect(("127.0.0.1", port_r1))

        pkt = Packet(
            version=1,
            src="sender-1",
            dest="receiver-1",
            payload="Weighted route delivery",
            packet_id="pkt-weighted-multihop",
            ttl=8,
        )
        send_frame(sender_sock, pkt.to_json_bytes())

        # Sender receives DELIVERED response from R1
        resp_bytes = recv_frame(sender_sock)
        self.assertIsNotNone(resp_bytes)
        resp = RouterResponse.from_json_bytes(resp_bytes)
        self.assertEqual(resp.status, "DELIVERED")
        self.assertEqual(resp.packet_id, "pkt-weighted-multihop")

        # Receiver receives packet via R2 with TTL decremented twice (8 -> 7 -> 6)
        conn, _ = recv_listener.accept()
        self.sockets.append(conn)
        conn.settimeout(3.0)
        rx_bytes = recv_frame(conn)
        self.assertIsNotNone(rx_bytes)
        rx_pkt = Packet.from_json_bytes(rx_bytes)
        self.assertEqual(rx_pkt.packet_id, "pkt-weighted-multihop")
        self.assertEqual(rx_pkt.payload, "Weighted route delivery")
        self.assertEqual(rx_pkt.src, "sender-1")
        self.assertEqual(rx_pkt.dest, "receiver-1")
        self.assertEqual(rx_pkt.ttl, 6)

        try:
            sender_sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        try:
            conn.shutdown(socket.SHUT_WR)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
