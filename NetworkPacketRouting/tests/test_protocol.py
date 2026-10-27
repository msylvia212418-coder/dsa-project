"""
test_protocol.py - Unit and socket tests for length-prefixed framing.

Verifies encoding, decoding, partial reads, stream boundaries, truncation,
size validation, clean connection termination, and resource cleanup.
"""

import socket
import struct
import unittest

from NetworkPacketRouting.protocol import (
    HEADER_FORMAT,
    HEADER_LENGTH,
    MAX_FRAME_SIZE,
    MIN_FRAME_SIZE,
    FrameSizeError,
    IncompleteFrameError,
    pack_header,
    read_exact,
    recv_frame,
    send_frame,
    unpack_header,
)


class TestProtocolHeader(unittest.TestCase):
    """Tests for frame header packing and unpacking logic."""

    def test_pack_and_unpack_valid_lengths(self) -> None:
        """Verify normal frame lengths pack to 4-byte big-endian and unpack accurately."""
        test_lengths = [1, 10, 1024, 65536, MAX_FRAME_SIZE]
        for length in test_lengths:
            packed = pack_header(length)
            self.assertEqual(len(packed), HEADER_LENGTH)
            # Verify network byte order (big-endian)
            expected_bytes = struct.pack(">I", length)
            self.assertEqual(packed, expected_bytes)

            unpacked = unpack_header(packed)
            self.assertEqual(unpacked, length)

    def test_pack_header_type_error(self) -> None:
        """Verify packing non-integers and booleans raises TypeError."""
        for invalid_val in (True, False, "10", 10.5, None, [10]):
            with self.assertRaises(TypeError, msg=f"Failed for {invalid_val!r}"):
                pack_header(invalid_val)  # type: ignore[arg-type]

    def test_pack_zero_length_rejected(self) -> None:
        """Verify packing a zero-length frame raises FrameSizeError."""
        with self.assertRaises(FrameSizeError) as ctx:
            pack_header(0)
        self.assertIn("at least 1", str(ctx.exception))

    def test_pack_negative_length_rejected(self) -> None:
        """Verify packing a negative length raises FrameSizeError."""
        with self.assertRaises(FrameSizeError):
            pack_header(-1)

    def test_pack_oversized_length_rejected(self) -> None:
        """Verify packing a length exceeding 1 MiB raises FrameSizeError."""
        with self.assertRaises(FrameSizeError) as ctx:
            pack_header(MAX_FRAME_SIZE + 1)
        self.assertIn("exceeds maximum", str(ctx.exception))

    def test_unpack_header_invalid_byte_length(self) -> None:
        """Verify unpacking header bytes not equal to 4 bytes raises IncompleteFrameError."""
        with self.assertRaises(IncompleteFrameError):
            unpack_header(b"\x00\x00")
        with self.assertRaises(IncompleteFrameError):
            unpack_header(b"\x00\x00\x00\x00\x01")

    def test_unpack_header_type_error(self) -> None:
        """Verify unpacking non-bytes header raises TypeError."""
        for invalid_header in ("1234", 1234, True, None):
            with self.assertRaises(TypeError, msg=f"Failed for {invalid_header!r}"):
                unpack_header(invalid_header)  # type: ignore[arg-type]

    def test_unpack_header_invalid_limit_arguments(self) -> None:
        """Verify configuring an invalid max_frame_size in unpack_header raises appropriate error."""
        valid_header = struct.pack(HEADER_FORMAT, 100)

        # Non-integer / boolean limits raise TypeError
        for invalid_limit in (True, False, "1024", 1024.5, None):
            with self.assertRaises(TypeError, msg=f"Failed for {invalid_limit!r}"):
                unpack_header(valid_header, max_frame_size=invalid_limit)  # type: ignore[arg-type]

        # Out-of-bounds limits raise ValueError
        for out_of_bounds in (0, -10, MAX_FRAME_SIZE + 1, MAX_FRAME_SIZE + 1000):
            with self.assertRaises(ValueError, msg=f"Failed for {out_of_bounds!r}"):
                unpack_header(valid_header, max_frame_size=out_of_bounds)

    def test_unpack_header_zero_or_oversized(self) -> None:
        """Verify unpacking header declaring 0 or >1 MiB raises FrameSizeError."""
        zero_header = struct.pack(HEADER_FORMAT, 0)
        with self.assertRaises(FrameSizeError):
            unpack_header(zero_header)

        oversized_header = struct.pack(HEADER_FORMAT, MAX_FRAME_SIZE + 100)
        with self.assertRaises(FrameSizeError):
            unpack_header(oversized_header)


class TestProtocolSocketFraming(unittest.TestCase):
    """Tests for socket framing, transmission, stream reassembly, and error conditions."""

    def setUp(self) -> None:
        """Create a connected local socket pair with deterministic timeouts."""
        self.client_sock, self.server_sock = socket.socketpair()
        self.client_sock.settimeout(2.0)
        self.server_sock.settimeout(2.0)
        self.addCleanup(self._cleanup_sockets)

    def _cleanup_sockets(self) -> None:
        """Ensure both sockets are closed after each test."""
        for sock in (self.client_sock, self.server_sock):
            try:
                sock.close()
            except OSError:
                pass

    def test_successful_frame_send_and_receive(self) -> None:
        """Verify a complete payload is transmitted and reassembled intact."""
        payload = b"Hello, Network Routing!"
        send_frame(self.client_sock, payload)

        received = recv_frame(self.server_sock)
        self.assertEqual(received, payload)

    def test_medium_payload_transmission(self) -> None:
        """Verify a 64 KB payload transmits and reassembles correctly."""
        payload = b"X" * (64 * 1024)
        send_frame(self.client_sock, payload)

        received = recv_frame(self.server_sock)
        self.assertEqual(received, payload)

    def test_multiple_sequential_frames_over_single_connection(self) -> None:
        """Verify multiple sequential frames over one persistent connection maintain FIFO order."""
        frames = [
            b"Frame 1: Initial Handshake",
            b"Frame 2: Intermediate Data Payload",
            b"Frame 3: Final Termination Notice",
        ]

        for frame in frames:
            send_frame(self.client_sock, frame)

        for expected in frames:
            received = recv_frame(self.server_sock)
            self.assertEqual(received, expected)

    def test_clean_eof_before_new_frame(self) -> None:
        """Verify recv_frame returns None when client disconnects cleanly between frames."""
        send_frame(self.client_sock, b"Single Message")
        self.client_sock.close()

        # First read receives the message
        first_read = recv_frame(self.server_sock)
        self.assertEqual(first_read, b"Single Message")

        # Second read detects clean EOF before any new header byte
        second_read = recv_frame(self.server_sock)
        self.assertIsNone(second_read)

    def test_immediate_clean_eof(self) -> None:
        """Verify recv_frame returns None if peer disconnects immediately without sending bytes."""
        self.client_sock.close()
        result = recv_frame(self.server_sock)
        self.assertIsNone(result)

    def test_partial_reads_and_fragmented_delivery(self) -> None:
        """Verify read_exact and recv_frame reassemble frames fragmented into 1-byte chunks."""
        payload = b"Fragmented Message Payload"
        header = pack_header(len(payload))
        full_stream = header + payload

        # Send byte-by-byte to simulate extreme network fragmentation
        for byte in full_stream:
            self.client_sock.sendall(bytes([byte]))

        received = recv_frame(self.server_sock)
        self.assertEqual(received, payload)

    def test_truncated_header_raises_incomplete_frame_error(self) -> None:
        """Verify client disconnect after sending only 2 header bytes raises IncompleteFrameError."""
        # Send only 2 of the 4 required header bytes, then close socket
        self.client_sock.sendall(b"\x00\x00")
        self.client_sock.close()

        with self.assertRaises(IncompleteFrameError) as ctx:
            recv_frame(self.server_sock)
        self.assertIn("Connection closed prematurely", str(ctx.exception))

    def test_truncated_body_raises_incomplete_frame_error(self) -> None:
        """Verify client disconnect after partial body transmission raises IncompleteFrameError."""
        # Header declares 10 bytes, but only 4 bytes are sent before socket close
        header = struct.pack(HEADER_FORMAT, 10)
        self.client_sock.sendall(header + b"1234")
        self.client_sock.close()

        with self.assertRaises(IncompleteFrameError) as ctx:
            recv_frame(self.server_sock)
        self.assertIn("received 4 of 10 expected bytes", str(ctx.exception))

    def test_zero_length_frame_rejected(self) -> None:
        """Verify sending or receiving an explicitly zero-length frame raises FrameSizeError."""
        with self.assertRaises(FrameSizeError):
            send_frame(self.client_sock, b"")

        # Simulate receiving a header claiming 0 bytes
        zero_header = struct.pack(HEADER_FORMAT, 0)
        self.client_sock.sendall(zero_header)

        with self.assertRaises(FrameSizeError) as ctx:
            recv_frame(self.server_sock)
        self.assertIn("Invalid frame size 0 bytes", str(ctx.exception))

    def test_oversized_frame_declaration_rejected(self) -> None:
        """Verify receiving a header declaring size > 1 MiB raises FrameSizeError without reading body."""
        oversized_header = struct.pack(HEADER_FORMAT, MAX_FRAME_SIZE + 50)
        self.client_sock.sendall(oversized_header)

        with self.assertRaises(FrameSizeError) as ctx:
            recv_frame(self.server_sock)
        self.assertIn("exceeds limit", str(ctx.exception))

    def test_recv_frame_invalid_limit_configuration(self) -> None:
        """Verify configuring max_frame_size > MAX_FRAME_SIZE, <=0, or non-int raises error."""
        # Non-integer / boolean raises TypeError
        with self.assertRaises(TypeError):
            recv_frame(self.server_sock, max_frame_size=True)  # type: ignore[arg-type]

        # Limit exceeding MAX_FRAME_SIZE raises ValueError
        with self.assertRaises(ValueError):
            recv_frame(self.server_sock, max_frame_size=MAX_FRAME_SIZE + 1)

        # Limit <= 0 raises ValueError
        with self.assertRaises(ValueError):
            recv_frame(self.server_sock, max_frame_size=0)

    def test_recv_frame_custom_valid_limit_enforced(self) -> None:
        """Verify a custom valid max_frame_size (e.g. 50 bytes) rejects a 60-byte frame header."""
        custom_header = struct.pack(HEADER_FORMAT, 60)
        self.client_sock.sendall(custom_header)

        with self.assertRaises(FrameSizeError) as ctx:
            recv_frame(self.server_sock, max_frame_size=50)
        self.assertIn("exceeds limit of 50 bytes", str(ctx.exception))

    def test_send_frame_type_error(self) -> None:
        """Verify passing a non-bytes object to send_frame raises TypeError."""
        with self.assertRaises(TypeError):
            send_frame(self.client_sock, "Not bytes, a string")  # type: ignore[arg-type]

    def test_read_exact_invalid_arguments(self) -> None:
        """Verify read_exact raises TypeError for non-int and ValueError if num_bytes is <= 0."""
        with self.assertRaises(TypeError):
            read_exact(self.server_sock, True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            read_exact(self.server_sock, 0)
        with self.assertRaises(ValueError):
            read_exact(self.server_sock, -5)


if __name__ == "__main__":
    unittest.main()
