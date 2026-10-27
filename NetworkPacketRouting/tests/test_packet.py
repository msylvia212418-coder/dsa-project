"""
test_packet.py - Unit tests for Packet and RouterResponse schemas and serialization.

Tests valid round trips, boundary conditions, type enforcement (including boolean rejection),
oversized payloads, malformed JSON, invalid UTF-8, unpaired surrogates, missing/unexpected fields,
and response statuses.
"""

import json
import unittest

from NetworkPacketRouting.packet import (
    MAX_DETAILS_LENGTH,
    MAX_IDENTIFIER_LENGTH,
    MAX_PAYLOAD_BYTES,
    SUPPORTED_VERSION,
    VALID_STATUSES,
    Packet,
    PacketValidationError,
    RouterResponse,
)


class TestPacketSchema(unittest.TestCase):
    """Unit tests for the Packet data model, validation, and JSON serialization."""

    def setUp(self) -> None:
        """Create a standard valid packet for use in tests."""
        self.valid_kwargs = {
            "version": 1,
            "src": "node-sender",
            "dest": "node-receiver",
            "packet_id": "pkt-1001",
            "ttl": 16,
            "payload": "Sample message payload",
        }

    def test_valid_packet_creation_and_roundtrip(self) -> None:
        """Verify a valid packet can be created, serialized to JSON bytes, and deserialized identically."""
        packet = Packet(**self.valid_kwargs)
        self.assertEqual(packet.version, 1)
        self.assertEqual(packet.src, "node-sender")
        self.assertEqual(packet.dest, "node-receiver")
        self.assertEqual(packet.packet_id, "pkt-1001")
        self.assertEqual(packet.ttl, 16)
        self.assertEqual(packet.payload, "Sample message payload")

        json_bytes = packet.to_json_bytes()
        self.assertIsInstance(json_bytes, bytes)

        restored = Packet.from_json_bytes(json_bytes)
        self.assertEqual(restored, packet)

    def test_valid_unicode_payloads(self) -> None:
        """Verify valid diverse Unicode characters (emojis, accented, CJK) work seamlessly."""
        unicode_payload = "Hello 🌍! Bonjour à tous, 这是一个网络路由测试。 (Accents: é, à, ü, ñ)"
        kwargs = dict(self.valid_kwargs, payload=unicode_payload)
        pkt = Packet(**kwargs)
        self.assertEqual(pkt.payload, unicode_payload)

        restored = Packet.from_json_bytes(pkt.to_json_bytes())
        self.assertEqual(restored.payload, unicode_payload)

    def test_unpaired_surrogate_in_direct_construction(self) -> None:
        """Verify direct Packet construction with an unpaired Unicode surrogate raises PacketValidationError."""
        unpaired_surrogates = ["\ud800", "\udfff", "Hello \ud83d World"]
        for bad_payload in unpaired_surrogates:
            kwargs = dict(self.valid_kwargs, payload=bad_payload)
            with self.assertRaises(PacketValidationError, msg=f"Allowed surrogate {bad_payload!r}") as ctx:
                Packet(**kwargs)
            self.assertIn("surrogates not allowed", str(ctx.exception))

    def test_unpaired_surrogate_in_json_deserialization(self) -> None:
        """Verify deserializing JSON with an escaped unpaired surrogate raises PacketValidationError."""
        # JSON standard allows \uD800 escape sequences, which Python decodes to surrogate strings
        bad_json_bytes = (
            b'{"version": 1, "src": "sender-1", "dest": "dest-1", '
            b'"packet_id": "pkt-1", "ttl": 10, "payload": "\\ud800"}'
        )
        with self.assertRaises(PacketValidationError) as ctx:
            Packet.from_json_bytes(bad_json_bytes)
        self.assertIn("surrogates not allowed", str(ctx.exception))

    def test_packet_version_strict_validation(self) -> None:
        """Verify version must be exactly integer 1, rejecting other ints, booleans, and types."""
        # Non-1 versions
        for invalid_v in (0, 2, -1, 10):
            kwargs = dict(self.valid_kwargs, version=invalid_v)
            with self.assertRaises(PacketValidationError):
                Packet(**kwargs)

        # Booleans and non-ints
        for invalid_type in (True, False, "1", 1.0, None):
            kwargs = dict(self.valid_kwargs, version=invalid_type)
            with self.assertRaises(PacketValidationError):
                Packet(**kwargs)

    def test_packet_ttl_strict_validation(self) -> None:
        """Verify TTL must be an integer >= 1, rejecting booleans, non-ints, and values < 1."""
        for invalid_ttl in (0, -1, -100):
            kwargs = dict(self.valid_kwargs, ttl=invalid_ttl)
            with self.assertRaises(PacketValidationError):
                Packet(**kwargs)

        for invalid_type in (True, False, "16", 16.5, None):
            kwargs = dict(self.valid_kwargs, ttl=invalid_type)
            with self.assertRaises(PacketValidationError):
                Packet(**kwargs)

    def test_packet_identifier_boundaries(self) -> None:
        """Verify src, dest, and packet_id enforce non-empty strings and length <= 64."""
        for field in ("src", "dest", "packet_id"):
            # Empty and whitespace-only
            for empty_val in ("", "   ", "\t\n"):
                kwargs = dict(self.valid_kwargs, **{field: empty_val})
                with self.assertRaises(PacketValidationError, msg=f"Allowed empty {field}"):
                    Packet(**kwargs)

            # Max length boundary: 64 characters should succeed
            valid_64 = "X" * MAX_IDENTIFIER_LENGTH
            kwargs = dict(self.valid_kwargs, **{field: valid_64})
            pkt = Packet(**kwargs)
            self.assertEqual(getattr(pkt, field), valid_64)

            # Length > 64 characters should fail
            invalid_65 = "X" * (MAX_IDENTIFIER_LENGTH + 1)
            kwargs = dict(self.valid_kwargs, **{field: invalid_65})
            with self.assertRaises(PacketValidationError, msg=f"Allowed 65-char {field}"):
                Packet(**kwargs)

            # Non-string types
            for non_str in (123, True, None, ["id"]):
                kwargs = dict(self.valid_kwargs, **{field: non_str})
                with self.assertRaises(PacketValidationError, msg=f"Allowed non-str {field}"):
                    Packet(**kwargs)

    def test_packet_payload_size_boundaries(self) -> None:
        """Verify payload allows up to 64 KiB of UTF-8 bytes and rejects larger payloads."""
        # Non-string payload
        for non_str in (123, True, None, b"bytes payload"):
            kwargs = dict(self.valid_kwargs, payload=non_str)
            with self.assertRaises(PacketValidationError):
                Packet(**kwargs)

        # Exact 64 KiB payload (65,536 ASCII bytes)
        payload_64k = "A" * MAX_PAYLOAD_BYTES
        pkt = Packet(**dict(self.valid_kwargs, payload=payload_64k))
        self.assertEqual(len(pkt.payload), MAX_PAYLOAD_BYTES)

        # 64 KiB + 1 byte
        payload_oversized = "A" * (MAX_PAYLOAD_BYTES + 1)
        with self.assertRaises(PacketValidationError) as ctx:
            Packet(**dict(self.valid_kwargs, payload=payload_oversized))
        self.assertIn("exceeds maximum limit", str(ctx.exception))

        # Multi-byte UTF-8 test (euro symbol '€' is 3 bytes in UTF-8)
        # 21,845 chars * 3 bytes = 65,535 bytes (<= 64 KiB)
        valid_multibyte = "€" * 21845
        pkt_multi = Packet(**dict(self.valid_kwargs, payload=valid_multibyte))
        self.assertEqual(pkt_multi.payload, valid_multibyte)

        # 21,846 chars * 3 bytes = 65,538 bytes (> 64 KiB)
        invalid_multibyte = "€" * 21846
        with self.assertRaises(PacketValidationError):
            Packet(**dict(self.valid_kwargs, payload=invalid_multibyte))

    def test_with_decremented_ttl(self) -> None:
        """Verify with_decremented_ttl decrements TTL by 1 and rejects TTL <= 1."""
        pkt = Packet(**dict(self.valid_kwargs, ttl=5))
        decremented = pkt.with_decremented_ttl()
        self.assertEqual(decremented.ttl, 4)
        self.assertEqual(decremented.packet_id, pkt.packet_id)
        self.assertEqual(decremented.payload, pkt.payload)

        # If TTL is 1, decrementing is prohibited
        pkt_ttl_1 = Packet(**dict(self.valid_kwargs, ttl=1))
        with self.assertRaises(PacketValidationError) as ctx:
            pkt_ttl_1.with_decremented_ttl()
        self.assertIn("has expired (TTL=1)", str(ctx.exception))

    def test_deserialization_missing_and_unexpected_fields(self) -> None:
        """Verify missing fields or extra fields raise PacketValidationError during deserialization."""
        # Test each missing field
        for field in ("version", "src", "dest", "packet_id", "ttl", "payload"):
            data = dict(self.valid_kwargs)
            del data[field]
            raw_bytes = json.dumps(data).encode("utf-8")
            with self.assertRaises(PacketValidationError, msg=f"Missing {field} not detected"):
                Packet.from_json_bytes(raw_bytes)

        # Test unexpected extra field
        data_extra = dict(self.valid_kwargs, extra_field="unauthorized")
        raw_bytes_extra = json.dumps(data_extra).encode("utf-8")
        with self.assertRaises(PacketValidationError) as ctx:
            Packet.from_json_bytes(raw_bytes_extra)
        self.assertIn("Unexpected field(s)", str(ctx.exception))

    def test_deserialization_invalid_format_and_types(self) -> None:
        """Verify malformed JSON, invalid UTF-8, non-dict roots, and non-bytes raise PacketValidationError."""
        # Malformed JSON syntax
        with self.assertRaises(PacketValidationError) as ctx:
            Packet.from_json_bytes(b'{"version": 1, "src": ')
        self.assertIn("Malformed JSON", str(ctx.exception))

        # Invalid UTF-8 bytes
        with self.assertRaises(PacketValidationError) as ctx:
            Packet.from_json_bytes(b"\xff\xfe\x00\x01")
        self.assertIn("not valid UTF-8", str(ctx.exception))

        # Non-dict JSON root (array or primitive)
        with self.assertRaises(PacketValidationError):
            Packet.from_json_bytes(b"[1, 2, 3]")
        with self.assertRaises(PacketValidationError):
            Packet.from_json_bytes(b'"A simple string"')

        # Non-bytes input
        with self.assertRaises(PacketValidationError):
            Packet.from_json_bytes("not bytes")  # type: ignore[arg-type]


class TestRouterResponseSchema(unittest.TestCase):
    """Unit tests for the RouterResponse feedback model, validation, and JSON serialization."""

    def setUp(self) -> None:
        self.valid_kwargs = {
            "version": 1,
            "packet_id": "pkt-1001",
            "status": "DELIVERED",
            "details": "Forwarded successfully to receiver-1",
        }

    def test_valid_statuses_roundtrip(self) -> None:
        """Verify all supported router status codes serialize and deserialize cleanly."""
        for status in VALID_STATUSES:
            resp = RouterResponse(
                version=1,
                packet_id="pkt-1001",
                status=status,
                details=f"Status reason for {status}",
            )
            self.assertEqual(resp.status, status)

            json_bytes = resp.to_json_bytes()
            restored = RouterResponse.from_json_bytes(json_bytes)
            self.assertEqual(restored, resp)

    def test_empty_details_allowed(self) -> None:
        """Verify details can be an empty string."""
        resp = RouterResponse(
            version=1,
            packet_id="pkt-1001",
            status="DELIVERED",
            details="",
        )
        self.assertEqual(resp.details, "")
        restored = RouterResponse.from_json_bytes(resp.to_json_bytes())
        self.assertEqual(restored.details, "")

    def test_invalid_status_rejected(self) -> None:
        """Verify unapproved status codes raise PacketValidationError."""
        for bad_status in ("SUCCESS", "FAILED", "ACK", "NACK", "UNKNOWN", ""):
            kwargs = dict(self.valid_kwargs, status=bad_status)
            with self.assertRaises(PacketValidationError):
                RouterResponse(**kwargs)

    def test_details_length_boundary(self) -> None:
        """Verify details allows up to 512 characters and rejects >512."""
        valid_details = "D" * MAX_DETAILS_LENGTH
        resp = RouterResponse(**dict(self.valid_kwargs, details=valid_details))
        self.assertEqual(len(resp.details), MAX_DETAILS_LENGTH)

        invalid_details = "D" * (MAX_DETAILS_LENGTH + 1)
        with self.assertRaises(PacketValidationError):
            RouterResponse(**dict(self.valid_kwargs, details=invalid_details))

    def test_response_missing_and_extra_fields(self) -> None:
        """Verify missing or unexpected fields in RouterResponse JSON are rejected."""
        # Missing field
        data = dict(self.valid_kwargs)
        del data["status"]
        with self.assertRaises(PacketValidationError):
            RouterResponse.from_json_bytes(json.dumps(data).encode("utf-8"))

        # Extra field
        data_extra = dict(self.valid_kwargs, unknown_key="extra")
        with self.assertRaises(PacketValidationError):
            RouterResponse.from_json_bytes(json.dumps(data_extra).encode("utf-8"))

    def test_response_invalid_types_and_booleans(self) -> None:
        """Verify non-integer or boolean versions in RouterResponse are rejected."""
        with self.assertRaises(PacketValidationError):
            RouterResponse(version=True, packet_id="p1", status="DELIVERED", details="")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
