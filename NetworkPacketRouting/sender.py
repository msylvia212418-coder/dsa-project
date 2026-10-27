"""
sender.py - Interactive TCP packet sender.

Connects to a Router (default: 127.0.0.1:5001), encapsulates user messages into
validated length-prefixed Packet frames, streams them over a persistent connection,
and displays synchronous RouterResponse feedback.
"""

from __future__ import annotations

import argparse
import socket
import sys
import uuid
from pathlib import Path

# Ensure root directory is on sys.path if run directly as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from NetworkPacketRouting.packet import (
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

DEFAULT_ROUTER_HOST = "127.0.0.1"
DEFAULT_ROUTER_PORT = 5001
DEFAULT_SRC = "sender-1"
DEFAULT_DEST = "receiver-1"
DEFAULT_TTL = 16


def parse_args() -> argparse.Namespace:
    """Parses optional command-line arguments for sender configuration."""
    parser = argparse.ArgumentParser(description="Network Packet Routing - Interactive Sender")
    parser.add_argument("--host", default=DEFAULT_ROUTER_HOST, help="Router host IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_ROUTER_PORT, help="Router port (default: 5001)")
    parser.add_argument("--src", default=DEFAULT_SRC, help="Source node ID (default: sender-1)")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="Destination node ID (default: receiver-1)")
    parser.add_argument("--ttl", type=int, default=DEFAULT_TTL, help="Initial packet TTL (default: 16)")
    return parser.parse_args()


def start_sender(
    host: str = DEFAULT_ROUTER_HOST,
    port: int = DEFAULT_ROUTER_PORT,
    src: str = DEFAULT_SRC,
    dest: str = DEFAULT_DEST,
    ttl: int = DEFAULT_TTL,
) -> None:
    """
    Connects to the router, maintains a persistent connection for multiple user messages,
    transmits length-prefixed packets, and prints synchronous router responses.
    """
    print("=" * 60)
    print(" NETWORK PACKET ROUTING - SENDER")
    print(f" Target Router : {host}:{port}")
    print(f" Route Path    : {src} -> {dest} (Initial TTL: {ttl})")
    print("=" * 60)

    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5.0)
        client.connect((host, port))
    except ConnectionRefusedError:
        print(f"\n[ERROR] Connection refused at {host}:{port}. Ensure router.py is running.")
        return
    except OSError as err:
        print(f"\n[ERROR] Could not connect to router at {host}:{port}: {err}")
        return

    print("Connected to router! Type a message and press Enter.")
    print("Type /quit or /exit to disconnect.\n")

    counter = 1
    with client:
        while True:
            try:
                message = input("You: ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting sender...")
                break

            if not message.strip():
                continue

            if message.strip().lower() in ("/quit", "/exit"):
                print("Disconnecting from router...")
                break

            packet_id = f"pkt-{uuid.uuid4().hex[:6]}-{counter}"
            counter += 1

            try:
                packet = Packet(
                    version=1,
                    src=src,
                    dest=dest,
                    packet_id=packet_id,
                    ttl=ttl,
                    payload=message,
                )
            except PacketValidationError as val_err:
                print(f"[LOCAL ERROR] Failed to construct packet: {val_err}")
                continue

            # Send framed packet over persistent socket
            try:
                send_frame(client, packet.to_json_bytes())
            except (OSError, FrameSizeError) as send_err:
                print(f"[ERROR] Failed to send frame to router: {send_err}")
                break

            # Read synchronous router feedback response
            try:
                response_bytes = recv_frame(client)
            except (IncompleteFrameError, FrameSizeError) as framing_err:
                print(f"[ERROR] Framing error reading router response: {framing_err}")
                break
            except (socket.timeout, OSError) as recv_err:
                print(f"[ERROR] Failed to read router response: {recv_err}")
                break

            if response_bytes is None:
                print("[WARN] Router closed the connection.")
                break

            try:
                response = RouterResponse.from_json_bytes(response_bytes)
                status_symbol = "[OK]" if response.status == "DELIVERED" else "[DROP/FAIL]"
                print(
                    f"  {status_symbol} Status: {response.status} | "
                    f"Packet ID: {response.packet_id}\n"
                    f"  Details: {response.details}\n"
                )
            except PacketValidationError as resp_err:
                print(f"[WARN] Received invalid router response: {resp_err}\n")

    print("Sender disconnected.")


if __name__ == "__main__":
    args = parse_args()
    try:
        start_sender(
            host=args.host,
            port=args.port,
            src=args.src,
            dest=args.dest,
            ttl=args.ttl,
        )
    except KeyboardInterrupt:
        print("\nSender stopped by user.")
