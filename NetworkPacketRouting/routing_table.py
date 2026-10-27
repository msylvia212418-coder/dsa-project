"""
routing_table.py - In-memory static routing table with O(1) hash map lookup.

This module provides the RoutingTable data structure that maps network destination
identifiers to (host, port) network tuples. It is decoupled from socket I/O.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

MIN_PORT: int = 1
MAX_PORT: int = 65535
DEFAULT_DESTINATION: str = "receiver-1"
DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 5002


class RoutingTableError(Exception):
    """Base exception for routing table validation and configuration errors."""
    pass


class InvalidRouteError(RoutingTableError):
    """Raised when destination, host, or port configuration is invalid."""
    pass


class RoutingTable:
    """
    Static routing table storing destination-to-address mappings.

    Provides average O(1) time complexity route lookups using an internal
    hash map (Python dictionary).
    """

    def __init__(self, populate_default: bool = True) -> None:
        """
        Initializes the routing table.

        Args:
            populate_default: If True, registers the planned default route:
                              'receiver-1' -> ('127.0.0.1', 5002).
        """
        self._routes: Dict[str, Tuple[str, int]] = {}
        if populate_default:
            self.add_route(DEFAULT_DESTINATION, DEFAULT_HOST, DEFAULT_PORT)

    def _validate_destination(self, destination: str) -> str:
        """Validates that a destination identifier is a non-empty string without leading/trailing whitespace."""
        if type(destination) is not str:
            raise InvalidRouteError(
                f"Destination must be a string, got {type(destination).__name__}"
            )
        if not destination:
            raise InvalidRouteError("Destination cannot be empty")
        if destination != destination.strip():
            raise InvalidRouteError("Destination cannot contain leading or trailing whitespace")
        return destination

    def _validate_host(self, host: str) -> str:
        """Validates that a host address is a non-empty string without leading/trailing whitespace."""
        if type(host) is not str:
            raise InvalidRouteError(
                f"Host must be a string, got {type(host).__name__}"
            )
        if not host:
            raise InvalidRouteError("Host cannot be empty")
        if host != host.strip():
            raise InvalidRouteError("Host cannot contain leading or trailing whitespace")
        return host

    def _validate_port(self, port: int) -> int:
        """Validates that a port is an integer in the range 1-65535, rejecting booleans."""
        if type(port) is not int or isinstance(port, bool):
            raise InvalidRouteError(
                f"Port must be an integer, got {type(port).__name__}"
            )
        if port < MIN_PORT or port > MAX_PORT:
            raise InvalidRouteError(
                f"Port must be between {MIN_PORT} and {MAX_PORT}, got {port}"
            )
        return port

    def add_route(self, destination: str, host: str, port: int) -> None:
        """
        Adds or updates a destination route in the routing table.

        Args:
            destination: Target node identifier.
            host: IP address or hostname of the destination.
            port: TCP port of the destination (1-65535).

        Raises:
            InvalidRouteError: If any argument fails validation.
        """
        dest_valid = self._validate_destination(destination)
        host_valid = self._validate_host(host)
        port_valid = self._validate_port(port)
        self._routes[dest_valid] = (host_valid, port_valid)

    def get_route(self, destination: str) -> Optional[Tuple[str, int]]:
        """
        Looks up the route for a destination.

        Args:
            destination: Target node identifier.

        Returns:
            A tuple of (host, port) if the route exists, or None if unknown.

        Raises:
            InvalidRouteError: If destination is not a string, empty, or has leading/trailing whitespace.
        """
        dest_valid = self._validate_destination(destination)
        return self._routes.get(dest_valid)

    def remove_route(self, destination: str) -> bool:
        """
        Removes a route from the table safely.

        Args:
            destination: Target node identifier to remove.

        Returns:
            True if the route was present and removed, False otherwise.

        Raises:
            InvalidRouteError: If destination is not a string, empty, or has leading/trailing whitespace.
        """
        dest_valid = self._validate_destination(destination)
        if dest_valid in self._routes:
            del self._routes[dest_valid]
            return True
        return False

    def has_route(self, destination: str) -> bool:
        """
        Checks if a route exists for the given destination.

        Args:
            destination: Target node identifier.

        Returns:
            True if route exists, False otherwise.
        """
        dest_valid = self._validate_destination(destination)
        return dest_valid in self._routes

    def __len__(self) -> int:
        """Returns the number of registered routes in the table."""
        return len(self._routes)

    def get_all_routes(self) -> Dict[str, Tuple[str, int]]:
        """Returns a copy of all registered routes."""
        return dict(self._routes)
