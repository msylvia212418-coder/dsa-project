"""
test_router.py - Unit and integration tests for the synchronous Router engine.

Tests router lifecycle, forwarding accuracy, TTL decrement, drop policies,
route lookup, downstream connection failures, invalid packet handling,
connection recovery, and sequential multi-packet sessions.
"""

import socket
import struct
import threading
import unittest
from typing import List, Optional, Tuple

from NetworkPacketRouting.packet import (
    Packet,
    RouterResponse,
)
from NetworkPacketRouting.protocol import (
    HEADER_FORMAT,
    recv_frame,
    send_frame,
)
from NetworkPacketRouting.router import (
    UNKNOWN_PACKET_ID,
    Router,
)
from NetworkPacketRouting.routing_table import RoutingTable


class TestRouterEngine(unittest.TestCase):
    """Integration test suite for the synchronous Router process."""

    def setUp(self) -> None:
        """Initialize tracking lists for clean resource disposal in teardown."""
        self.routers: List[Router] = []
        self.router_threads: List[threading.Thread] = []
        self.sockets: List[socket.socket] = []

    def tearDown(self) -> None:
        """Ensure all sockets, destination servers, and router threads terminate cleanly."""
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

        for thread in self.router_threads:
            thread.join(timeout=2.0)

    def _start_router(self, routing_table: Optional[RoutingTable] = None) -> Tuple[Router, threading.Thread]:
        """Helper to boot a Router instance on an OS-assigned ephemeral loopback port."""
        router = Router(
            host="127.0.0.1",
            port=0,
            routing_table=routing_table,
            downstream_timeout=1.0,
        )
        router.start()
        self.routers.append(router)

        thread = threading.Thread(target=router.serve_forever, daemon=True)
        thread.start()
        self.router_threads.append(thread)
        return router, thread

    def _create_client_connection(self, router: Router) -> socket.socket:
        """Helper to establish a timed TCP client socket connection to the router."""
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.settimeout(2.0)
        client_sock.connect(("127.0.0.1", router.server_port))
        self.sockets.append(client_sock)
        return client_sock

    def test_router_starts_and_stops_cleanly(self) -> None:
        """Verify the router starts, binds an ephemeral port, and server thread terminates on stop()."""
        router, thread = self._start_router()
        self.assertTrue(router.is_running)
        self.assertTrue(thread.is_alive())
        self.assertGreater(router.server_port, 0)

        router.stop()
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive(), "Router server thread failed to terminate after stop()")
        self.assertFalse(router.is_running)

    def test_successful_packet_forwarding(self) -> None:
        """Verify valid packet is forwarded with decremented TTL and returns DELIVERED."""
        # 1. Set up a dummy destination server on an ephemeral port
        dest_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        dest_server.bind(("127.0.0.1", 0))
        dest_server.listen(1)
        dest_server.settimeout(2.0)
        self.sockets.append(dest_server)
        dest_port = dest_server.getsockname()[1]

        # 2. Configure routing table pointing to destination server
        table = RoutingTable(populate_default=False)
        table.add_route("dest-node", "127.0.0.1", dest_port)

        # 3. Start router and client
        router, _ = self._start_router(routing_table=table)
        client = self._create_client_connection(router)

        # 4. Client sends packet with TTL=5
        outbound = Packet(
            version=1,
            src="sender-node",
            dest="dest-node",
            packet_id="pkt-forward-1",
            ttl=5,
            payload="Forwarding Test Message",
        )
        send_frame(client, outbound.to_json_bytes())

        # 5. Destination accepts forwarded connection and reads packet
        dest_conn, _ = dest_server.accept()
        self.sockets.append(dest_conn)
        dest_conn.settimeout(2.0)

        forwarded_raw = recv_frame(dest_conn)
        self.assertIsNotNone(forwarded_raw)
        assert forwarded_raw is not None
        forwarded_packet = Packet.from_json_bytes(forwarded_raw)

        # Verify forwarded fields and TTL decrement by 1
        self.assertEqual(forwarded_packet.version, 1)
        self.assertEqual(forwarded_packet.src, "sender-node")
        self.assertEqual(forwarded_packet.dest, "dest-node")
        self.assertEqual(forwarded_packet.packet_id, "pkt-forward-1")
        self.assertEqual(forwarded_packet.ttl, 4)
        self.assertEqual(forwarded_packet.payload, "Forwarding Test Message")

        # 6. Client reads synchronous router feedback response
        response_raw = recv_frame(client)
        self.assertIsNotNone(response_raw)
        assert response_raw is not None
        response = RouterResponse.from_json_bytes(response_raw)

        self.assertEqual(response.version, 1)
        self.assertEqual(response.packet_id, "pkt-forward-1")
        self.assertEqual(response.status, "DELIVERED")
        self.assertIn("forwarded successfully", response.details)
        self.assertIn("not an end-to-end", response.details)

    def test_ttl_expiry_returns_dropped_ttl_without_contacting_destination(self) -> None:
        """Verify packet with TTL=1 is dropped with DROPPED_TTL and never forwarded."""
        dest_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        dest_server.bind(("127.0.0.1", 0))
        dest_server.listen(1)
        dest_server.settimeout(0.3)
        self.sockets.append(dest_server)
        dest_port = dest_server.getsockname()[1]

        table = RoutingTable(populate_default=False)
        table.add_route("dest-node", "127.0.0.1", dest_port)

        router, _ = self._start_router(routing_table=table)
        client = self._create_client_connection(router)

        # Send packet with TTL=1
        outbound = Packet(
            version=1,
            src="sender-node",
            dest="dest-node",
            packet_id="pkt-ttl-1",
            ttl=1,
            payload="TTL Boundary Payload",
        )
        send_frame(client, outbound.to_json_bytes())

        # Receive router response
        response_raw = recv_frame(client)
        self.assertIsNotNone(response_raw)
        assert response_raw is not None
        response = RouterResponse.from_json_bytes(response_raw)

        self.assertEqual(response.status, "DROPPED_TTL")
        self.assertEqual(response.packet_id, "pkt-ttl-1")
        self.assertIn("TTL expired", response.details)

        # Destination should not have received any connection
        with self.assertRaises(socket.timeout):
            dest_server.accept()

    def test_unknown_destination_returns_no_route(self) -> None:
        """Verify unknown destination ID returns NO_ROUTE without crashing."""
        table = RoutingTable(populate_default=False)
        table.add_route("known-dest", "127.0.0.1", 5000)

        router, _ = self._start_router(routing_table=table)
        client = self._create_client_connection(router)

        outbound = Packet(
            version=1,
            src="sender-node",
            dest="unregistered-dest",
            packet_id="pkt-no-route",
            ttl=10,
            payload="Unroutable Payload",
        )
        send_frame(client, outbound.to_json_bytes())

        response_raw = recv_frame(client)
        self.assertIsNotNone(response_raw)
        assert response_raw is not None
        response = RouterResponse.from_json_bytes(response_raw)

        self.assertEqual(response.status, "NO_ROUTE")
        self.assertEqual(response.packet_id, "pkt-no-route")
        self.assertIn("No route found", response.details)

    def test_downstream_unreachable_returns_unreachable(self) -> None:
        """Verify connection failure to destination returns UNREACHABLE without router crash."""
        # Pick an unused port by briefly binding and immediately closing
        temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        temp_sock.bind(("127.0.0.1", 0))
        offline_port = temp_sock.getsockname()[1]
        temp_sock.close()

        table = RoutingTable(populate_default=False)
        table.add_route("offline-node", "127.0.0.1", offline_port)

        router, _ = self._start_router(routing_table=table)
        client = self._create_client_connection(router)

        outbound = Packet(
            version=1,
            src="sender-node",
            dest="offline-node",
            packet_id="pkt-offline",
            ttl=10,
            payload="Payload to offline node",
        )
        send_frame(client, outbound.to_json_bytes())

        response_raw = recv_frame(client)
        self.assertIsNotNone(response_raw)
        assert response_raw is not None
        response = RouterResponse.from_json_bytes(response_raw)

        self.assertEqual(response.status, "UNREACHABLE")
        self.assertEqual(response.packet_id, "pkt-offline")
        self.assertIn("unreachable", response.details.lower())

    def test_malformed_json_returns_invalid_packet(self) -> None:
        """Verify non-JSON frame returns INVALID_PACKET with safe placeholder ID."""
        router, _ = self._start_router()
        client = self._create_client_connection(router)

        send_frame(client, b"THIS IS NOT VALID JSON")

        response_raw = recv_frame(client)
        self.assertIsNotNone(response_raw)
        assert response_raw is not None
        response = RouterResponse.from_json_bytes(response_raw)

        self.assertEqual(response.status, "INVALID_PACKET")
        self.assertEqual(response.packet_id, UNKNOWN_PACKET_ID)
        self.assertIn("validation failed", response.details.lower())

    def test_valid_packet_after_invalid_frame_on_same_connection(self) -> None:
        """Verify that after receiving an invalid packet frame, router stays alive and processes next packet."""
        # Set up destination
        dest_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        dest_server.bind(("127.0.0.1", 0))
        dest_server.listen(1)
        dest_server.settimeout(2.0)
        self.sockets.append(dest_server)
        dest_port = dest_server.getsockname()[1]

        table = RoutingTable(populate_default=False)
        table.add_route("dest-node", "127.0.0.1", dest_port)

        router, _ = self._start_router(routing_table=table)
        client = self._create_client_connection(router)

        # 1. Send schema-invalid packet (missing ttl, but includes packet_id)
        bad_json = b'{"version": 1, "src": "s", "dest": "d", "packet_id": "pkt-bad", "payload": "hi"}'
        send_frame(client, bad_json)

        resp1_raw = recv_frame(client)
        self.assertIsNotNone(resp1_raw)
        assert resp1_raw is not None
        resp1 = RouterResponse.from_json_bytes(resp1_raw)
        self.assertEqual(resp1.status, "INVALID_PACKET")
        self.assertEqual(resp1.packet_id, "pkt-bad")

        # 2. Send valid packet over the exact same socket connection
        valid_packet = Packet(
            version=1,
            src="s",
            dest="dest-node",
            packet_id="pkt-good-after-bad",
            ttl=4,
            payload="I am valid!",
        )
        send_frame(client, valid_packet.to_json_bytes())

        # Destination receives the forwarded valid packet
        dest_conn, _ = dest_server.accept()
        self.sockets.append(dest_conn)
        forwarded_raw = recv_frame(dest_conn)
        self.assertIsNotNone(forwarded_raw)
        assert forwarded_raw is not None
        forwarded = Packet.from_json_bytes(forwarded_raw)
        self.assertEqual(forwarded.packet_id, "pkt-good-after-bad")
        self.assertEqual(forwarded.ttl, 3)

        # Client receives DELIVERED feedback for second packet
        resp2_raw = recv_frame(client)
        self.assertIsNotNone(resp2_raw)
        assert resp2_raw is not None
        resp2 = RouterResponse.from_json_bytes(resp2_raw)
        self.assertEqual(resp2.status, "DELIVERED")
        self.assertEqual(resp2.packet_id, "pkt-good-after-bad")

    def test_surrogate_in_packet_id_returns_unknown_packet_id_and_recovers(self) -> None:
        """Verify escaped unpaired surrogate in packet_id yields UNKNOWN_PACKET_ID and connection recovers."""
        dest_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        dest_server.bind(("127.0.0.1", 0))
        dest_server.listen(1)
        dest_server.settimeout(2.0)
        self.sockets.append(dest_server)
        dest_port = dest_server.getsockname()[1]

        table = RoutingTable(populate_default=False)
        table.add_route("dest-node", "127.0.0.1", dest_port)

        router, _ = self._start_router(routing_table=table)
        client = self._create_client_connection(router)

        # 1. Send JSON containing an escaped surrogate in packet_id
        surrogate_json = (
            b'{"version": 1, "src": "src-node", "dest": "dest-node", '
            b'"packet_id": "\\ud800", "ttl": 5, "payload": "hi"}'
        )
        send_frame(client, surrogate_json)

        resp1_raw = recv_frame(client)
        self.assertIsNotNone(resp1_raw)
        assert resp1_raw is not None
        resp1 = RouterResponse.from_json_bytes(resp1_raw)
        self.assertEqual(resp1.status, "INVALID_PACKET")
        self.assertEqual(resp1.packet_id, UNKNOWN_PACKET_ID)

        # 2. Verify connection remains healthy for subsequent valid packet
        subsequent = Packet(
            version=1,
            src="src-node",
            dest="dest-node",
            packet_id="pkt-recovered",
            ttl=6,
            payload="Healthy message",
        )
        send_frame(client, subsequent.to_json_bytes())

        dest_conn, _ = dest_server.accept()
        self.sockets.append(dest_conn)
        dest_fwd = recv_frame(dest_conn)
        self.assertIsNotNone(dest_fwd)
        assert dest_fwd is not None
        fwd_pkt = Packet.from_json_bytes(dest_fwd)
        self.assertEqual(fwd_pkt.packet_id, "pkt-recovered")
        self.assertEqual(fwd_pkt.ttl, 5)

        resp2_raw = recv_frame(client)
        self.assertIsNotNone(resp2_raw)
        assert resp2_raw is not None
        resp2 = RouterResponse.from_json_bytes(resp2_raw)
        self.assertEqual(resp2.status, "DELIVERED")
        self.assertEqual(resp2.packet_id, "pkt-recovered")

    def test_whitespace_in_packet_id_falls_back_to_unknown_packet_id(self) -> None:
        """Verify leading/trailing whitespace in packet_id is rejected by salvage logic."""
        router, _ = self._start_router()
        client = self._create_client_connection(router)

        bad_id_json = b'{"version": 1, "src": "s", "dest": "d", "packet_id": "  padded-id  ", "payload": "hi"}'
        send_frame(client, bad_id_json)

        resp_raw = recv_frame(client)
        self.assertIsNotNone(resp_raw)
        assert resp_raw is not None
        resp = RouterResponse.from_json_bytes(resp_raw)
        self.assertEqual(resp.status, "INVALID_PACKET")
        self.assertEqual(resp.packet_id, UNKNOWN_PACKET_ID)

    def test_multiple_sequential_packets_over_persistent_connection(self) -> None:
        """Verify multiple packets sent sequentially receive correlated responses in FIFO order."""
        dest_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        dest_server.bind(("127.0.0.1", 0))
        dest_server.listen(5)
        dest_server.settimeout(2.0)
        self.sockets.append(dest_server)
        dest_port = dest_server.getsockname()[1]

        table = RoutingTable(populate_default=False)
        table.add_route("dest-seq", "127.0.0.1", dest_port)

        router, _ = self._start_router(routing_table=table)
        client = self._create_client_connection(router)

        num_packets = 4
        for i in range(num_packets):
            pkt_id = f"pkt-seq-{i}"
            outbound = Packet(
                version=1,
                src="sender-seq",
                dest="dest-seq",
                packet_id=pkt_id,
                ttl=10,
                payload=f"Payload sequence number {i}",
            )
            send_frame(client, outbound.to_json_bytes())

            # Destination accepts connection and reads forwarded frame
            conn, _ = dest_server.accept()
            self.sockets.append(conn)
            fwd_raw = recv_frame(conn)
            self.assertIsNotNone(fwd_raw)
            assert fwd_raw is not None
            fwd_packet = Packet.from_json_bytes(fwd_raw)
            self.assertEqual(fwd_packet.packet_id, pkt_id)
            self.assertEqual(fwd_packet.ttl, 9)

            # Client receives corresponding feedback
            resp_raw = recv_frame(client)
            self.assertIsNotNone(resp_raw)
            assert resp_raw is not None
            resp = RouterResponse.from_json_bytes(resp_raw)
            self.assertEqual(resp.status, "DELIVERED")
            self.assertEqual(resp.packet_id, pkt_id)

    def test_truncated_frame_closes_sender_connection(self) -> None:
        """Verify truncated frame body causes router to close connection without hanging."""
        router, _ = self._start_router()
        client = self._create_client_connection(router)

        # Header declares 20 bytes, but we only send 5 bytes then shut down writing
        header = struct.pack(HEADER_FORMAT, 20)
        client.sendall(header + b"12345")
        client.shutdown(socket.SHUT_WR)

        # The router should detect truncation and close the socket; reading from client returns EOF
        eof_check = client.recv(1024)
        self.assertEqual(eof_check, b"")

    def test_update_routing_table_valid(self) -> None:
        """Verify update_routing_table replaces the router's active table."""
        router = Router(host="127.0.0.1", port=0)
        initial_table = router.routing_table
        self.assertIsInstance(initial_table, RoutingTable)

        new_table = RoutingTable(populate_default=False)
        new_table.add_route("custom-node", "127.0.0.1", 9999)

        router.update_routing_table(new_table)
        self.assertIs(router.routing_table, new_table)
        self.assertTrue(router.routing_table.has_route("custom-node"))

    def test_update_routing_table_invalid_raises(self) -> None:
        """Verify update_routing_table raises TypeError for non-RoutingTable argument."""
        router = Router(host="127.0.0.1", port=0)
        with self.assertRaises(TypeError):
            router.update_routing_table("not-a-table")  # type: ignore
        with self.assertRaises(TypeError):
            router.update_routing_table(None)  # type: ignore
        with self.assertRaises(TypeError):
            router.update_routing_table({})  # type: ignore


if __name__ == "__main__":
    unittest.main()
