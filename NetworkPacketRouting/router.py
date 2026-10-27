"""
router.py - Synchronous single-router TCP packet forwarding engine.

Listens for incoming sender connections, accepts length-prefixed packet frames,
validates schemas, enforces Time-To-Live (TTL) mechanics, looks up destinations
in the RoutingTable, forwards valid packets downstream, and returns synchronous
framed RouterResponse feedback for each processed frame.
"""

from __future__ import annotations

import json
import logging
import socket
from typing import Optional, Set, Tuple

from NetworkPacketRouting.packet import (
    MAX_DETAILS_LENGTH,
    Packet,
    PacketValidationError,
    RouterResponse,
)
from NetworkPacketRouting.protocol import (
    FrameSizeError,
    IncompleteFrameError,
    recv_frame,
    send_frame,
)
from NetworkPacketRouting.routing_table import RoutingTable

DEFAULT_ROUTER_HOST: str = "0.0.0.0"
DEFAULT_ROUTER_PORT: int = 5001
DOWNSTREAM_TIMEOUT: float = 2.0
ACCEPT_TIMEOUT: float = 0.2
UNKNOWN_PACKET_ID: str = "unknown-packet"

logger = logging.getLogger("NetworkPacketRouting.router")


class Router:
    """
    Synchronous TCP Router that forwards framed packets based on a RoutingTable.

    Lifecycle:
        router = Router(host='127.0.0.1', port=0)
        router.start()
        # In main or worker thread:
        router.serve_forever()
        ...
        router.stop()
    """

    def __init__(
        self,
        host: str = DEFAULT_ROUTER_HOST,
        port: int = DEFAULT_ROUTER_PORT,
        routing_table: Optional[RoutingTable] = None,
        downstream_timeout: float = DOWNSTREAM_TIMEOUT,
    ) -> None:
        """
        Initializes the router configuration.

        Args:
            host: Interface address to bind to (e.g. '0.0.0.0' or '127.0.0.1').
            port: Port to listen on (0 for dynamic OS assignment).
            routing_table: Table containing destination mappings (defaults to RoutingTable()).
            downstream_timeout: Timeout in seconds for connecting and sending to destinations.
        """
        self._host: str = host
        self._port: int = port
        self._routing_table: RoutingTable = routing_table if routing_table is not None else RoutingTable()
        self._downstream_timeout: float = downstream_timeout

        self._server_sock: Optional[socket.socket] = None
        self._assigned_port: Optional[int] = None
        self._running: bool = False
        self._active_clients: Set[socket.socket] = set()

    @property
    def server_port(self) -> int:
        """Returns the actual bound port (useful when initialized with port=0)."""
        if self._assigned_port is None:
            raise RuntimeError("Router has not been started yet")
        return self._assigned_port

    @property
    def is_running(self) -> bool:
        """Returns True if the router is actively serving."""
        return self._running

    def start(self) -> None:
        """Binds the server socket, sets options, and prepares to listen."""
        if self._server_sock is not None:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.listen(5)
        sock.settimeout(ACCEPT_TIMEOUT)

        self._server_sock = sock
        self._assigned_port = sock.getsockname()[1]
        self._running = True
        logger.info("Router started listening on %s:%d", self._host, self._assigned_port)

    def serve_forever(self) -> None:
        """
        Accepts connections and processes them synchronously until stop() is called.
        """
        if self._server_sock is None:
            self.start()

        assert self._server_sock is not None
        while self._running:
            try:
                client_sock, client_addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if not self._running:
                    break
                logger.exception("Unexpected error in router accept loop")
                continue

            # Process client connection synchronously
            self._handle_client(client_sock, client_addr)

    def _handle_client(self, client_sock: socket.socket, client_addr: Tuple[str, int]) -> None:
        """
        Handles multiple sequential framed packets over a persistent sender connection.
        """
        self._active_clients.add(client_sock)
        logger.debug("Sender connected from %s:%d", client_addr[0], client_addr[1])

        try:
            while self._running:
                try:
                    payload_bytes = recv_frame(client_sock)
                except (IncompleteFrameError, FrameSizeError) as framing_err:
                    logger.warning(
                        "Framing error from sender %s:%d: %s. Terminating connection.",
                        client_addr[0],
                        client_addr[1],
                        framing_err,
                    )
                    break
                except OSError as sock_err:
                    logger.warning(
                        "Socket error reading from %s:%d: %s. Terminating connection.",
                        client_addr[0],
                        client_addr[1],
                        sock_err,
                    )
                    break

                # Clean EOF: sender closed connection gracefully between frames
                if payload_bytes is None:
                    logger.debug("Sender %s:%d disconnected cleanly", client_addr[0], client_addr[1])
                    break

                # Process the complete frame and obtain feedback response
                response = self._process_frame(payload_bytes)

                # Return framed response to sender
                try:
                    send_frame(client_sock, response.to_json_bytes())
                except OSError as send_err:
                    logger.warning(
                        "Failed to send feedback to %s:%d: %s. Terminating connection.",
                        client_addr[0],
                        client_addr[1],
                        send_err,
                    )
                    break
        finally:
            self._active_clients.discard(client_sock)
            try:
                client_sock.close()
            except OSError:
                pass

    def _process_frame(self, payload_bytes: bytes) -> RouterResponse:
        """
        Processes a single validly-framed packet payload according to router rules.
        """
        # 1. Deserialize and validate packet schema
        try:
            packet = Packet.from_json_bytes(payload_bytes)
        except PacketValidationError as val_err:
            packet_id = self._salvage_packet_id(payload_bytes)
            details = f"Packet validation failed: {val_err}"[:MAX_DETAILS_LENGTH]
            logger.warning("Packet validation error for packet_id=%s: %s", packet_id, val_err)
            try:
                return RouterResponse(
                    version=1,
                    packet_id=packet_id,
                    status="INVALID_PACKET",
                    details=details,
                )
            except Exception:
                return RouterResponse(
                    version=1,
                    packet_id=UNKNOWN_PACKET_ID,
                    status="INVALID_PACKET",
                    details="Packet validation failed",
                )

        # 2. Check TTL: If TTL <= 1, packet cannot survive another forwarding hop
        if packet.ttl <= 1:
            details = f"Packet {packet.packet_id} dropped: TTL expired (TTL={packet.ttl})"
            logger.info(details)
            return RouterResponse(
                version=1,
                packet_id=packet.packet_id,
                status="DROPPED_TTL",
                details=details[:MAX_DETAILS_LENGTH],
            )

        # 3. Route lookup in RoutingTable
        route = self._routing_table.get_route(packet.dest)
        if route is None:
            details = f"Packet {packet.packet_id} dropped: No route found for destination '{packet.dest}'"
            logger.info(details)
            return RouterResponse(
                version=1,
                packet_id=packet.packet_id,
                status="NO_ROUTE",
                details=details[:MAX_DETAILS_LENGTH],
            )

        dest_host, dest_port = route

        # 4. Decrement TTL for forwarding
        decremented_packet = packet.with_decremented_ttl()

        # 5. Connect downstream and forward packet
        downstream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        downstream_sock.settimeout(self._downstream_timeout)

        try:
            downstream_sock.connect((dest_host, dest_port))
            send_frame(downstream_sock, decremented_packet.to_json_bytes())
        except (OSError, socket.timeout) as net_err:
            details = (
                f"Packet {packet.packet_id} unreachable: Failed to forward to {packet.dest} "
                f"at {dest_host}:{dest_port} ({net_err})"
            )[:MAX_DETAILS_LENGTH]
            logger.warning(details)
            return RouterResponse(
                version=1,
                packet_id=packet.packet_id,
                status="UNREACHABLE",
                details=details,
            )
        finally:
            try:
                downstream_sock.close()
            except OSError:
                pass

        # 6. Forwarding succeeded
        details = (
            f"Packet {packet.packet_id} forwarded to {packet.dest} at {dest_host}:{dest_port}; "
            f"forwarded successfully (not an end-to-end application acknowledgement)"
        )[:MAX_DETAILS_LENGTH]
        logger.info("Successfully forwarded packet %s to %s", packet.packet_id, packet.dest)
        return RouterResponse(
            version=1,
            packet_id=packet.packet_id,
            status="DELIVERED",
            details=details,
        )

    def _salvage_packet_id(self, payload_bytes: bytes) -> str:
        """
        Attempts to extract a valid, non-empty packet_id from raw bytes if JSON was parsed,
        falling back to the safe placeholder UNKNOWN_PACKET_ID.
        """
        try:
            raw_text = payload_bytes.decode("utf-8")
            raw_dict = json.loads(raw_text)
            if isinstance(raw_dict, dict) and type(raw_dict.get("packet_id")) is str:
                candidate = raw_dict["packet_id"]
                # Must be non-empty, without leading/trailing whitespace, and <= 64 chars
                if candidate and candidate == candidate.strip() and len(candidate) <= 64:
                    # Verify string encodes to valid UTF-8 without surrogates
                    candidate.encode("utf-8")
                    return candidate
        except Exception:
            pass
        return UNKNOWN_PACKET_ID

    def stop(self) -> None:
        """
        Signals the router to stop and unblocks any waiting threads or connections.
        """
        self._running = False

        # Close all active client connections
        for client in list(self._active_clients):
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass
        self._active_clients.clear()

        # Unblock accept() immediately by connecting a dummy loopback socket if bound
        if self._server_sock is not None:
            bound_port = self._assigned_port
            if bound_port is not None:
                try:
                    dummy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    dummy.settimeout(0.5)
                    dummy.connect(("127.0.0.1", bound_port))
                    dummy.close()
                except OSError:
                    pass

            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None

        logger.info("Router stopped")
