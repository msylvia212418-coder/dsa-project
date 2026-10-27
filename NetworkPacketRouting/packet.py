"""
packet.py - Structured packet and router response schemas and validation.

This module provides data models for network packets and synchronous router
responses with strict schema validation, JSON UTF-8 serialization, and TTL mechanics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Set

# Schema Constants
SUPPORTED_VERSION: int = 1
MAX_IDENTIFIER_LENGTH: int = 64
MAX_PAYLOAD_BYTES: int = 64 * 1024  # 64 KiB = 65,536 bytes
MAX_DETAILS_LENGTH: int = 512

VALID_STATUSES: Set[str] = {
    "DELIVERED",
    "DROPPED_TTL",
    "NO_ROUTE",
    "UNREACHABLE",
    "INVALID_PACKET",
}

PACKET_REQUIRED_FIELDS: Set[str] = {
    "version",
    "src",
    "dest",
    "packet_id",
    "ttl",
    "payload",
}

RESPONSE_REQUIRED_FIELDS: Set[str] = {
    "version",
    "packet_id",
    "status",
    "details",
}


class PacketValidationError(Exception):
    """Raised when packet or router response schema validation fails."""
    pass


def _validate_str(value: Any, field_name: str, max_len: int, allow_empty: bool = False) -> str:
    """Validates that a field is a string within length bounds and valid UTF-8."""
    if type(value) is not str:
        raise PacketValidationError(
            f"Field '{field_name}' must be a string, got {type(value).__name__}"
        )
    if not allow_empty and not value.strip():
        raise PacketValidationError(
            f"Field '{field_name}' cannot be empty or contain only whitespace"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PacketValidationError(
            f"Field '{field_name}' contains invalid Unicode characters (surrogates not allowed): {exc}"
        ) from exc
    if len(value) > max_len:
        raise PacketValidationError(
            f"Field '{field_name}' exceeds maximum length of {max_len} characters (length: {len(value)})"
        )
    return value


def _validate_int(value: Any, field_name: str, min_val: int, expected_exact: int | None = None) -> int:
    """Validates that a field is a strict integer (rejecting booleans) and meets value bounds."""
    if type(value) is not int or isinstance(value, bool):
        raise PacketValidationError(
            f"Field '{field_name}' must be an integer, got {type(value).__name__}"
        )
    if expected_exact is not None and value != expected_exact:
        raise PacketValidationError(
            f"Field '{field_name}' must be {expected_exact}, got {value}"
        )
    if value < min_val:
        raise PacketValidationError(
            f"Field '{field_name}' must be at least {min_val}, got {value}"
        )
    return value


@dataclass(frozen=True)
class Packet:
    """
    Represents a structured network packet navigating the network.

    Fields:
        version: Protocol version, strictly 1.
        src: Source node ID (1-64 chars).
        dest: Destination node ID (1-64 chars).
        packet_id: Unique packet identifier (1-64 chars).
        ttl: Time To Live (integer >= 1).
        payload: Message body (maximum 64 KiB when encoded in UTF-8).
    """
    version: int
    src: str
    dest: str
    packet_id: str
    ttl: int
    payload: str

    def __post_init__(self) -> None:
        """Enforce strict field validation upon instance construction."""
        _validate_int(self.version, "version", min_val=1, expected_exact=SUPPORTED_VERSION)
        _validate_str(self.src, "src", max_len=MAX_IDENTIFIER_LENGTH)
        _validate_str(self.dest, "dest", max_len=MAX_IDENTIFIER_LENGTH)
        _validate_str(self.packet_id, "packet_id", max_len=MAX_IDENTIFIER_LENGTH)
        _validate_int(self.ttl, "ttl", min_val=1)

        if type(self.payload) is not str:
            raise PacketValidationError(
                f"Field 'payload' must be a string, got {type(self.payload).__name__}"
            )
        try:
            encoded = self.payload.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise PacketValidationError(
                f"Field 'payload' contains invalid Unicode characters (surrogates not allowed): {exc}"
            ) from exc

        if len(encoded) > MAX_PAYLOAD_BYTES:
            raise PacketValidationError(
                f"Payload size ({len(encoded)} bytes) exceeds maximum limit of {MAX_PAYLOAD_BYTES} bytes (64 KiB)"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the packet to a standard dictionary."""
        return {
            "version": self.version,
            "src": self.src,
            "dest": self.dest,
            "packet_id": self.packet_id,
            "ttl": self.ttl,
            "payload": self.payload,
        }

    def to_json_bytes(self) -> bytes:
        """Serializes the packet into UTF-8 encoded JSON bytes."""
        try:
            return json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise PacketValidationError(
                f"Failed to encode packet to UTF-8 JSON: {exc}"
            ) from exc

    def with_decremented_ttl(self) -> Packet:
        """
        Creates a new Packet instance with TTL decremented by 1.

        Raises:
            PacketValidationError: If current TTL <= 1, as it cannot be forwarded further.
        """
        if self.ttl <= 1:
            raise PacketValidationError(
                f"Cannot decrement TTL: packet {self.packet_id} has expired (TTL={self.ttl})"
            )
        return Packet(
            version=self.version,
            src=self.src,
            dest=self.dest,
            packet_id=self.packet_id,
            ttl=self.ttl - 1,
            payload=self.payload,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Packet:
        """Validates and creates a Packet from a raw dictionary."""
        if not isinstance(data, dict):
            raise PacketValidationError(f"Packet data must be a dictionary, got {type(data).__name__}")

        keys = set(data.keys())
        missing = PACKET_REQUIRED_FIELDS - keys
        if missing:
            raise PacketValidationError(f"Missing required packet field(s): {sorted(missing)}")

        extra = keys - PACKET_REQUIRED_FIELDS
        if extra:
            raise PacketValidationError(f"Unexpected field(s) in packet: {sorted(extra)}")

        return cls(
            version=data["version"],
            src=data["src"],
            dest=data["dest"],
            packet_id=data["packet_id"],
            ttl=data["ttl"],
            payload=data["payload"],
        )

    @classmethod
    def from_json_bytes(cls, data: bytes) -> Packet:
        """
        Deserializes a Packet from UTF-8 encoded JSON bytes.

        Raises:
            PacketValidationError: If bytes cannot be decoded, JSON is malformed,
                                   fields are missing or unexpected, or validation fails.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise PacketValidationError(
                f"Data to deserialize must be bytes or bytearray, got {type(data).__name__}"
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PacketValidationError(f"Packet bytes are not valid UTF-8: {exc}") from exc

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PacketValidationError(f"Malformed JSON in packet: {exc}") from exc

        return cls.from_dict(parsed)


@dataclass(frozen=True)
class RouterResponse:
    """
    Represents a synchronous status response returned by the router to the sender.

    Fields:
        version: Protocol version, strictly 1.
        packet_id: Identifier of the packet this feedback refers to.
        status: Delivery or drop status code.
        details: Human-readable diagnostic details.
    """
    version: int
    packet_id: str
    status: str
    details: str

    def __post_init__(self) -> None:
        """Enforce strict field validation upon instance construction."""
        _validate_int(self.version, "version", min_val=1, expected_exact=SUPPORTED_VERSION)
        _validate_str(self.packet_id, "packet_id", max_len=MAX_IDENTIFIER_LENGTH)
        _validate_str(self.status, "status", max_len=32)

        if self.status not in VALID_STATUSES:
            raise PacketValidationError(
                f"Invalid router response status '{self.status}'. Must be one of: {sorted(VALID_STATUSES)}"
            )

        _validate_str(self.details, "details", max_len=MAX_DETAILS_LENGTH, allow_empty=True)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the response to a standard dictionary."""
        return {
            "version": self.version,
            "packet_id": self.packet_id,
            "status": self.status,
            "details": self.details,
        }

    def to_json_bytes(self) -> bytes:
        """Serializes the response into UTF-8 encoded JSON bytes."""
        try:
            return json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise PacketValidationError(
                f"Failed to encode response to UTF-8 JSON: {exc}"
            ) from exc

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RouterResponse:
        """Validates and creates a RouterResponse from a dictionary."""
        if not isinstance(data, dict):
            raise PacketValidationError(f"Response data must be a dictionary, got {type(data).__name__}")

        keys = set(data.keys())
        missing = RESPONSE_REQUIRED_FIELDS - keys
        if missing:
            raise PacketValidationError(f"Missing required response field(s): {sorted(missing)}")

        extra = keys - RESPONSE_REQUIRED_FIELDS
        if extra:
            raise PacketValidationError(f"Unexpected field(s) in response: {sorted(extra)}")

        return cls(
            version=data["version"],
            packet_id=data["packet_id"],
            status=data["status"],
            details=data["details"],
        )

    @classmethod
    def from_json_bytes(cls, data: bytes) -> RouterResponse:
        """
        Deserializes a RouterResponse from UTF-8 encoded JSON bytes.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise PacketValidationError(
                f"Data to deserialize must be bytes or bytearray, got {type(data).__name__}"
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PacketValidationError(f"Response bytes are not valid UTF-8: {exc}") from exc

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PacketValidationError(f"Malformed JSON in response: {exc}") from exc

        return cls.from_dict(parsed)
