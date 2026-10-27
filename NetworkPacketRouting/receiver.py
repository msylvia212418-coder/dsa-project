"""
receiver.py - Destination node server receiving forwarded network packets.

Listens on 0.0.0.0:5002 by default, validates incoming length-prefixed packet frames,
and logs received payload data and hop-by-hop metadata (remaining TTL).
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

# Ensure root directory is on sys.path if run directly as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from NetworkPacketRouting.packet import (
    Packet,
    PacketValidationError,
)
from NetworkPacketRouting.protocol import (
    FrameSizeError,
    IncompleteFrameError,
    recv_frame,
)

DEFAULT_RECEIVER_HOST = "0.0.0.0"
DEFAULT_RECEIVER_PORT = 5002


def parse_args() -> argparse.Namespace:
    """Parses optional command-line arguments for receiver configuration."""
    parser = argparse.ArgumentParser(description="Network Packet Routing - Receiver Node")
    parser.add_argument("--host", default=DEFAULT_RECEIVER_HOST, help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_RECEIVER_PORT, help="Port to listen on (default: 5002)")
    return parser.parse_args()


def start_receiver(host: str = DEFAULT_RECEIVER_HOST, port: int = DEFAULT_RECEIVER_PORT) -> None:
    """
    Starts the receiver server socket, accepts sequential connections, reads framed
    packets, and logs message details.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind((host, port))
        server.listen(5)
    except OSError as bind_err:
        print(f"[ERROR] Failed to bind receiver on {host}:{port}: {bind_err}", flush=True)
        return

    print("=" * 60, flush=True)
    print(" NETWORK PACKET ROUTING - RECEIVER NODE", flush=True)
    print(f" Listening on {host}:{port}...", flush=True)
    print(" Waiting for forwarded packets from Router. Press Ctrl+C to stop.", flush=True)
    print("=" * 60, flush=True)

    try:
        while True:
            try:
                conn, address = server.accept()
            except KeyboardInterrupt:
                break
            except OSError as accept_err:
                print(f"[ERROR] Accept failed: {accept_err}", flush=True)
                continue

            with conn:
                while True:
                    try:
                        payload_bytes = recv_frame(conn)
                    except (IncompleteFrameError, FrameSizeError) as framing_err:
                        print(f"[WARN] Framing error from {address[0]}:{address[1]}: {framing_err}", flush=True)
                        break
                    except OSError as sock_err:
                        print(f"[WARN] Socket error reading from {address[0]}:{address[1]}: {sock_err}", flush=True)
                        break

                    # Clean EOF: router finished forwarding frame(s) and closed connection
                    if payload_bytes is None:
                        break

                    try:
                        packet = Packet.from_json_bytes(payload_bytes)
                        print("-" * 50, flush=True)
                        print(f" [PACKET RECEIVED] ID: {packet.packet_id}", flush=True)
                        print(f" Source        : {packet.src}", flush=True)
                        print(f" Destination   : {packet.dest}", flush=True)
                        print(f" Remaining TTL : {packet.ttl}", flush=True)
                        print(f" Payload       : {packet.payload}", flush=True)
                        print("-" * 50, flush=True)
                    except PacketValidationError as val_err:
                        print(f"[WARN] Received invalid packet from {address[0]}:{address[1]}: {val_err}", flush=True)

    except KeyboardInterrupt:
        print("\nStopping receiver...", flush=True)
    finally:
        try:
            server.close()
        except OSError:
            pass
        print("Receiver stopped.", flush=True)


if __name__ == "__main__":
    args = parse_args()
    start_receiver(host=args.host, port=args.port)
