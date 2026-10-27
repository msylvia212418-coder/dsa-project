"""
protocol.py - Length-prefixed framing and stream I/O helpers.

This module provides robust TCP framing using a 4-byte big-endian unsigned
integer length prefix (>I). It ensures complete frame delivery across TCP's
unstructured byte stream, enforces maximum frame size limits (1 MiB), and
cleanly distinguishes between a normal EOF and a truncated/corrupted frame.
"""

from __future__ import annotations

import socket
import struct
from typing import Optional

# Framing Constants
HEADER_LENGTH: int = 4
HEADER_FORMAT: str = ">I"  # 4-byte unsigned big-endian integer
MAX_FRAME_SIZE: int = 1 * 1024 * 1024  # 1 MiB = 1,048,576 bytes
MIN_FRAME_SIZE: int = 1


class ProtocolError(Exception):
    """Base exception for all protocol and framing errors."""
    pass


class FrameSizeError(ProtocolError):
    """Raised when a frame length is zero, negative, or exceeds the allowable limit."""
    pass


class IncompleteFrameError(ProtocolError):
    """Raised when a socket connection closes prematurely while reading a frame."""
    pass


def _validate_limit(limit_value: int, param_name: str) -> None:
    """Validates that a size limit is a valid integer between MIN_FRAME_SIZE and MAX_FRAME_SIZE."""
    if type(limit_value) is not int or isinstance(limit_value, bool):
        raise TypeError(f"{param_name} must be an integer, got {type(limit_value).__name__}")
    if limit_value < MIN_FRAME_SIZE or limit_value > MAX_FRAME_SIZE:
        raise ValueError(
            f"{param_name} must be between {MIN_FRAME_SIZE} and {MAX_FRAME_SIZE} bytes, got {limit_value}"
        )


def pack_header(payload_length: int) -> bytes:
    """
    Packs a frame payload length into a 4-byte big-endian header.

    Args:
        payload_length: The length of the payload in bytes (must be an integer).

    Returns:
        4 bytes containing the encoded length prefix.

    Raises:
        TypeError: If payload_length is not an int (or is a boolean).
        FrameSizeError: If payload_length is < MIN_FRAME_SIZE or > MAX_FRAME_SIZE.
    """
    if type(payload_length) is not int or isinstance(payload_length, bool):
        raise TypeError(
            f"payload_length must be an integer, got {type(payload_length).__name__}"
        )
    if payload_length < MIN_FRAME_SIZE:
        raise FrameSizeError(
            f"Frame size must be at least {MIN_FRAME_SIZE} byte(s), got {payload_length}"
        )
    if payload_length > MAX_FRAME_SIZE:
        raise FrameSizeError(
            f"Frame size {payload_length} bytes exceeds maximum limit of {MAX_FRAME_SIZE} bytes (1 MiB)"
        )
    return struct.pack(HEADER_FORMAT, payload_length)


def unpack_header(header_bytes: bytes, max_frame_size: int = MAX_FRAME_SIZE) -> int:
    """
    Unpacks a 4-byte big-endian header into an integer payload length.

    Args:
        header_bytes: Exactly 4 bytes representing the frame length.
        max_frame_size: Maximum allowable frame size (cannot exceed MAX_FRAME_SIZE).

    Returns:
        The decoded integer payload length.

    Raises:
        TypeError: If header_bytes is not bytes/bytearray or max_frame_size is not an int.
        ValueError: If max_frame_size is not within [MIN_FRAME_SIZE, MAX_FRAME_SIZE].
        IncompleteFrameError: If header_bytes is not exactly 4 bytes.
        FrameSizeError: If the unpacked length is < MIN_FRAME_SIZE or > max_frame_size.
    """
    if not isinstance(header_bytes, (bytes, bytearray)):
        raise TypeError(
            f"header_bytes must be bytes or bytearray, got {type(header_bytes).__name__}"
        )
    _validate_limit(max_frame_size, "max_frame_size")

    if len(header_bytes) != HEADER_LENGTH:
        raise IncompleteFrameError(
            f"Header requires exactly {HEADER_LENGTH} bytes, received {len(header_bytes)}"
        )

    length: int = struct.unpack(HEADER_FORMAT, header_bytes)[0]
    if length < MIN_FRAME_SIZE:
        raise FrameSizeError(
            f"Invalid frame size {length} bytes in header (minimum is {MIN_FRAME_SIZE})"
        )
    if length > max_frame_size:
        raise FrameSizeError(
            f"Declared frame size {length} bytes exceeds limit of {max_frame_size} bytes"
        )
    return length


def read_exact(sock: socket.socket, num_bytes: int, allow_clean_eof: bool = False) -> bytes:
    """
    Reads exactly `num_bytes` from a socket, handling fragmented reads in a loop.

    Args:
        sock: The connected TCP socket to read from.
        num_bytes: The exact number of bytes required (must be a positive int).
        allow_clean_eof: If True and 0 bytes have been read when the socket reaches
                         EOF, returns empty bytes `b""` without error.
                         If False and EOF is reached at any point, raises IncompleteFrameError.

    Returns:
        bytes: Exactly `num_bytes` if read successfully, or `b""` if EOF occurred
               with 0 bytes read and `allow_clean_eof=True`.

    Raises:
        TypeError: If num_bytes is not an int or is a boolean.
        ValueError: If num_bytes <= 0.
        IncompleteFrameError: If the connection closes prematurely before num_bytes are read.
        OSError: If an underlying socket read error occurs.
    """
    if type(num_bytes) is not int or isinstance(num_bytes, bool):
        raise TypeError(f"num_bytes must be an integer, got {type(num_bytes).__name__}")
    if num_bytes <= 0:
        raise ValueError(f"num_bytes must be strictly positive, got {num_bytes}")

    buffer = bytearray()
    while len(buffer) < num_bytes:
        chunk = sock.recv(num_bytes - len(buffer))
        if not chunk:
            if allow_clean_eof and len(buffer) == 0:
                return b""
            raise IncompleteFrameError(
                f"Connection closed prematurely: received {len(buffer)} of {num_bytes} expected bytes"
            )
        buffer.extend(chunk)

    return bytes(buffer)


def send_frame(sock: socket.socket, payload: bytes) -> None:
    """
    Sends a complete length-prefixed frame over the given socket.

    Args:
        sock: The connected TCP socket to transmit over.
        payload: The raw bytes to be framed and sent.

    Raises:
        TypeError: If payload is not bytes or bytearray.
        FrameSizeError: If len(payload) is zero or exceeds MAX_FRAME_SIZE.
        OSError: If an underlying socket send error occurs.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"Payload must be bytes or bytearray, got {type(payload).__name__}")

    header = pack_header(len(payload))
    sock.sendall(header + payload)


def recv_frame(sock: socket.socket, max_frame_size: int = MAX_FRAME_SIZE) -> Optional[bytes]:
    """
    Receives a single length-prefixed frame from a socket.

    Args:
        sock: The connected TCP socket to read from.
        max_frame_size: Maximum allowable payload size (cannot exceed MAX_FRAME_SIZE).

    Returns:
        bytes: The decoded payload bytes.
        None: If the peer closed the connection cleanly before a new frame started.

    Raises:
        TypeError: If max_frame_size is not an integer or is a boolean.
        ValueError: If max_frame_size is <= 0 or exceeds MAX_FRAME_SIZE.
        FrameSizeError: If the declared frame size is zero or exceeds max_frame_size.
        IncompleteFrameError: If the connection terminates during header or body transmission.
        OSError: If an underlying socket read error occurs.
    """
    _validate_limit(max_frame_size, "max_frame_size")

    # 1. Read the 4-byte header (allowing clean EOF before any bytes are read)
    header = read_exact(sock, HEADER_LENGTH, allow_clean_eof=True)
    if header == b"":
        return None  # Clean connection closure between frames

    # 2. Decode the header and enforce frame limits using unpack_header
    payload_length = unpack_header(header, max_frame_size=max_frame_size)

    # 3. Read exactly payload_length bytes for the frame body (no clean EOF permitted mid-frame)
    return read_exact(sock, payload_length, allow_clean_eof=False)
