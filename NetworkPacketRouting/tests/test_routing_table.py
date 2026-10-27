"""
test_routing_table.py - Unit tests for the RoutingTable data structure.

Tests default routes, addition, updating, lookup, safe removal, port boundaries,
and strict type and value rejection (including leading/trailing whitespace rejection).
"""

import unittest

from NetworkPacketRouting.routing_table import (
    DEFAULT_DESTINATION,
    DEFAULT_HOST,
    DEFAULT_PORT,
    InvalidRouteError,
    RoutingTable,
)


class TestRoutingTable(unittest.TestCase):
    """Unit tests for RoutingTable functionality and input validation."""

    def test_default_route_initialization(self) -> None:
        """Verify the routing table initializes with the planned default route."""
        table = RoutingTable()
        self.assertEqual(len(table), 1)
        self.assertTrue(table.has_route(DEFAULT_DESTINATION))

        route = table.get_route(DEFAULT_DESTINATION)
        self.assertEqual(route, (DEFAULT_HOST, DEFAULT_PORT))

    def test_empty_initialization(self) -> None:
        """Verify initializing without default route produces an empty table."""
        table = RoutingTable(populate_default=False)
        self.assertEqual(len(table), 0)
        self.assertFalse(table.has_route(DEFAULT_DESTINATION))
        self.assertIsNone(table.get_route(DEFAULT_DESTINATION))

    def test_add_and_lookup_route(self) -> None:
        """Verify adding new routes succeeds and lookups return the expected (host, port) tuple."""
        table = RoutingTable(populate_default=False)
        table.add_route("node-alpha", "192.168.1.10", 6000)
        table.add_route("node-beta", "192.168.1.20", 7000)

        self.assertEqual(len(table), 2)
        self.assertEqual(table.get_route("node-alpha"), ("192.168.1.10", 6000))
        self.assertEqual(table.get_route("node-beta"), ("192.168.1.20", 7000))

    def test_update_existing_route(self) -> None:
        """Verify adding an existing destination updates its address and port without increasing size."""
        table = RoutingTable()
        self.assertEqual(len(table), 1)

        # Update the default destination with new host and port
        table.add_route(DEFAULT_DESTINATION, "10.0.0.5", 9999)
        self.assertEqual(len(table), 1)
        self.assertEqual(table.get_route(DEFAULT_DESTINATION), ("10.0.0.5", 9999))

    def test_unsuccessful_lookup_returns_none(self) -> None:
        """Verify looking up an unknown destination returns None without error."""
        table = RoutingTable()
        self.assertIsNone(table.get_route("nonexistent-node"))
        self.assertFalse(table.has_route("nonexistent-node"))

    def test_remove_route_existing_destination(self) -> None:
        """Verify removing an existing route returns True and deletes the route."""
        table = RoutingTable()
        self.assertTrue(table.has_route(DEFAULT_DESTINATION))

        removed = table.remove_route(DEFAULT_DESTINATION)
        self.assertTrue(removed)
        self.assertFalse(table.has_route(DEFAULT_DESTINATION))
        self.assertIsNone(table.get_route(DEFAULT_DESTINATION))
        self.assertEqual(len(table), 0)

    def test_remove_route_nonexistent_destination(self) -> None:
        """Verify removing an unknown destination returns False safely without raising exceptions."""
        table = RoutingTable()
        initial_len = len(table)

        removed = table.remove_route("unknown-target")
        self.assertFalse(removed)
        self.assertEqual(len(table), initial_len)

    def test_get_all_routes_isolation(self) -> None:
        """Verify get_all_routes returns a shallow copy that cannot mutate internal state."""
        table = RoutingTable()
        routes_copy = table.get_all_routes()
        self.assertIn(DEFAULT_DESTINATION, routes_copy)

        # Mutate the copy
        routes_copy["malicious-entry"] = ("0.0.0.0", 1111)
        self.assertFalse(table.has_route("malicious-entry"))
        self.assertEqual(len(table), 1)

    def test_destination_leading_trailing_whitespace_rejection(self) -> None:
        """Verify destinations with leading or trailing whitespace are explicitly rejected."""
        table = RoutingTable()
        bad_destinations = (" receiver-1", "receiver-1 ", "  node-a  ", "\tnode-b", "node-c\n")

        for bad_dest in bad_destinations:
            with self.assertRaises(InvalidRouteError) as ctx:
                table.add_route(bad_dest, "127.0.0.1", 5000)
            self.assertIn("leading or trailing whitespace", str(ctx.exception))

            with self.assertRaises(InvalidRouteError):
                table.get_route(bad_dest)

            with self.assertRaises(InvalidRouteError):
                table.remove_route(bad_dest)

            with self.assertRaises(InvalidRouteError):
                table.has_route(bad_dest)

    def test_host_leading_trailing_whitespace_rejection(self) -> None:
        """Verify hosts with leading or trailing whitespace are explicitly rejected."""
        table = RoutingTable()
        bad_hosts = (" 127.0.0.1", "127.0.0.1 ", "  192.168.1.1  ", "\tlocalhost", "localhost\n")

        for bad_host in bad_hosts:
            with self.assertRaises(InvalidRouteError) as ctx:
                table.add_route("valid-dest", bad_host, 5000)
            self.assertIn("leading or trailing whitespace", str(ctx.exception))

    def test_invalid_destination_rejection(self) -> None:
        """Verify non-string or empty destinations are rejected with InvalidRouteError."""
        table = RoutingTable()

        with self.assertRaises(InvalidRouteError) as ctx:
            table.add_route("", "127.0.0.1", 5000)
        self.assertIn("cannot be empty", str(ctx.exception))

        for invalid_type in (None, 123, True, False, ["dest"]):
            with self.assertRaises(InvalidRouteError, msg=f"Allowed non-str dest: {invalid_type!r}"):
                table.add_route(invalid_type, "127.0.0.1", 5000)  # type: ignore[arg-type]
            with self.assertRaises(InvalidRouteError):
                table.get_route(invalid_type)  # type: ignore[arg-type]
            with self.assertRaises(InvalidRouteError):
                table.remove_route(invalid_type)  # type: ignore[arg-type]

    def test_invalid_host_rejection(self) -> None:
        """Verify non-string or empty hosts are rejected with InvalidRouteError."""
        table = RoutingTable()

        with self.assertRaises(InvalidRouteError) as ctx:
            table.add_route("valid-dest", "", 5000)
        self.assertIn("cannot be empty", str(ctx.exception))

        for invalid_type in (None, 123, True, False, ["localhost"]):
            with self.assertRaises(InvalidRouteError, msg=f"Allowed non-str host: {invalid_type!r}"):
                table.add_route("valid-dest", invalid_type, 5000)  # type: ignore[arg-type]

    def test_port_type_rejection(self) -> None:
        """Verify non-integer and boolean port types are rejected with InvalidRouteError."""
        table = RoutingTable()

        # Booleans and non-ints
        for invalid_port in (True, False, "5000", 5000.5, None, [5000]):
            with self.assertRaises(InvalidRouteError, msg=f"Allowed invalid port: {invalid_port!r}"):
                table.add_route("valid-dest", "127.0.0.1", invalid_port)  # type: ignore[arg-type]

    def test_port_boundary_conditions(self) -> None:
        """Verify valid port boundaries (1 and 65535) and rejection of out-of-bounds ports."""
        table = RoutingTable(populate_default=False)

        # Valid boundaries: 1 and 65535
        table.add_route("min-port-node", "127.0.0.1", 1)
        self.assertEqual(table.get_route("min-port-node"), ("127.0.0.1", 1))

        table.add_route("max-port-node", "127.0.0.1", 65535)
        self.assertEqual(table.get_route("max-port-node"), ("127.0.0.1", 65535))

        # Underflow (< 1)
        for underflow_port in (0, -1, -5000):
            with self.assertRaises(InvalidRouteError, msg=f"Allowed underflow port {underflow_port}"):
                table.add_route("bad-port", "127.0.0.1", underflow_port)

        # Overflow (> 65535)
        for overflow_port in (65536, 70000, 100000):
            with self.assertRaises(InvalidRouteError, msg=f"Allowed overflow port {overflow_port}"):
                table.add_route("bad-port", "127.0.0.1", overflow_port)


if __name__ == "__main__":
    unittest.main()
