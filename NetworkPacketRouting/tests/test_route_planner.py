"""
test_route_planner.py - Unit tests for build_routing_table function.

Verifies:
- Linear topology next-hop assignment (all multi-hop routes use the first hop).
- Diamond topology deterministic tie-breaking based on edge insertion order.
- Direct neighbor routes matching their own endpoints.
- Isolated source nodes producing empty routing tables.
- Disconnected components/nodes being omitted from forwarding tables.
- Source node itself having no self-route entry.
- Rejection of invalid and unregistered source nodes with documented exceptions.
- Rejection of invalid topology objects.
- First-hop endpoint correctness (never using intermediate or final destination endpoints).
- Isolation of returned RoutingTable and preservation of topology state.
- Dynamic route recalculation upon link disconnection and reconnection.
"""

from __future__ import annotations

import unittest

from NetworkPacketRouting.route_planner import (
    RoutePlannerError,
    build_routing_table,
    build_weighted_routing_table,
)
from NetworkPacketRouting.routing_table import RoutingTable
from NetworkPacketRouting.topology import (
    InvalidNodeError,
    NetworkTopology,
    NodeNotFoundError,
)


class TestRoutePlannerValidation(unittest.TestCase):
    """Tests for argument validation in build_routing_table."""

    def setUp(self):
        self.topo = NetworkTopology()
        self.topo.register_node("R1", "127.0.0.1", 5001)

    def test_invalid_topology_type_raises(self):
        """Verify passing a non-NetworkTopology object raises RoutePlannerError."""
        with self.assertRaises(RoutePlannerError):
            build_routing_table("not-a-topology", "R1")  # type: ignore
        with self.assertRaises(RoutePlannerError):
            build_routing_table(None, "R1")  # type: ignore
        with self.assertRaises(RoutePlannerError):
            build_routing_table({}, "R1")  # type: ignore

    def test_unregistered_source_raises_node_not_found(self):
        """Verify unregistered source node raises NodeNotFoundError."""
        with self.assertRaises(NodeNotFoundError):
            build_routing_table(self.topo, "ghost-router")

    def test_invalid_source_identifiers_raise_invalid_node_error(self):
        """Verify invalid source node identifiers raise InvalidNodeError."""
        invalid_sources = [
            None,
            123,
            True,
            False,
            "",
            "   ",
            " R1",
            "R1 ",
            "\tR1",
            "R1\n",
        ]
        for bad_source in invalid_sources:
            with self.subTest(bad_source=bad_source):
                with self.assertRaises(InvalidNodeError):
                    build_routing_table(self.topo, bad_source)  # type: ignore


class TestRoutePlannerLinearTopology(unittest.TestCase):
    """Tests for routing tables compiled over a linear chain."""

    def test_linear_chain_next_hop_propagation(self):
        """
        Topology: S(5001) - R1(5002) - R2(5003) - D(5004)

        Expected routes from S:
        - R1 -> (127.0.0.1, 5002) [R1 endpoint]
        - R2 -> (127.0.0.1, 5002) [R1 endpoint]
        - D  -> (127.0.0.1, 5002) [R1 endpoint]

        Crucially: S must forward all traffic to R1, NOT to D's port (5004).
        """
        topo = NetworkTopology()
        topo.register_node("S", "127.0.0.1", 5001)
        topo.register_node("R1", "127.0.0.1", 5002)
        topo.register_node("R2", "127.0.0.1", 5003)
        topo.register_node("D", "127.0.0.1", 5004)

        topo.connect("S", "R1")
        topo.connect("R1", "R2")
        topo.connect("R2", "D")

        table = build_routing_table(topo, "S")

        self.assertIsInstance(table, RoutingTable)
        self.assertEqual(len(table), 3)

        # All multi-hop routes must point to the first hop (R1)
        self.assertEqual(table.get_route("R1"), ("127.0.0.1", 5002))
        self.assertEqual(table.get_route("R2"), ("127.0.0.1", 5002))
        self.assertEqual(table.get_route("D"), ("127.0.0.1", 5002))

        # Source node itself must not have a route
        self.assertFalse(table.has_route("S"))
        self.assertIsNone(table.get_route("S"))

    def test_linear_chain_from_interior_node(self):
        """
        From R1 in: S(5001) - R1(5002) - R2(5003) - D(5004)
        - Route to S must point to S (5001)
        - Route to R2 must point to R2 (5003)
        - Route to D must point to R2 (5003)
        """
        topo = NetworkTopology()
        topo.register_node("S", "127.0.0.1", 5001)
        topo.register_node("R1", "127.0.0.1", 5002)
        topo.register_node("R2", "127.0.0.1", 5003)
        topo.register_node("D", "127.0.0.1", 5004)

        topo.connect("S", "R1")
        topo.connect("R1", "R2")
        topo.connect("R2", "D")

        table = build_routing_table(topo, "R1")

        self.assertEqual(len(table), 3)
        self.assertEqual(table.get_route("S"), ("127.0.0.1", 5001))
        self.assertEqual(table.get_route("R2"), ("127.0.0.1", 5003))
        self.assertEqual(table.get_route("D"), ("127.0.0.1", 5003))
        self.assertFalse(table.has_route("R1"))


class TestRoutePlannerDiamondTopology(unittest.TestCase):
    """Tests for deterministic tie-breaking on diamond topologies."""

    def test_diamond_tie_breaking_order_a_first(self):
        """
        Topology:
              S
             / \\
            A   B
             \\ /
              D

        S connected to A first (5002), then B (5003).
        Route to D must point to A (5002).
        """
        topo = NetworkTopology()
        topo.register_node("S", "127.0.0.1", 5001)
        topo.register_node("A", "127.0.0.1", 5002)
        topo.register_node("B", "127.0.0.1", 5003)
        topo.register_node("D", "127.0.0.1", 5004)

        topo.connect("S", "A")  # A inserted first
        topo.connect("S", "B")
        topo.connect("A", "D")
        topo.connect("B", "D")

        table = build_routing_table(topo, "S")

        self.assertEqual(table.get_route("A"), ("127.0.0.1", 5002))
        self.assertEqual(table.get_route("B"), ("127.0.0.1", 5003))
        # D reached via A
        self.assertEqual(table.get_route("D"), ("127.0.0.1", 5002))

    def test_diamond_tie_breaking_order_b_first(self):
        """
        Same topology, but S connected to B first (5003), then A (5002).
        Route to D must point to B (5003).
        """
        topo = NetworkTopology()
        topo.register_node("S", "127.0.0.1", 5001)
        topo.register_node("A", "127.0.0.1", 5002)
        topo.register_node("B", "127.0.0.1", 5003)
        topo.register_node("D", "127.0.0.1", 5004)

        topo.connect("S", "B")  # B inserted first
        topo.connect("S", "A")
        topo.connect("B", "D")
        topo.connect("A", "D")

        table = build_routing_table(topo, "S")

        self.assertEqual(table.get_route("A"), ("127.0.0.1", 5002))
        self.assertEqual(table.get_route("B"), ("127.0.0.1", 5003))
        # D reached via B
        self.assertEqual(table.get_route("D"), ("127.0.0.1", 5003))


class TestRoutePlannerDirectDestinations(unittest.TestCase):
    """Tests for direct single-hop topologies."""

    def test_star_topology_direct_routes(self):
        """
        Star topology: Hub connected directly to Leaf1, Leaf2, Leaf3.
        Every destination route must point to that leaf's own endpoint.
        """
        topo = NetworkTopology()
        topo.register_node("Hub", "127.0.0.1", 5000)
        topo.register_node("Leaf1", "127.0.0.1", 5001)
        topo.register_node("Leaf2", "127.0.0.1", 5002)
        topo.register_node("Leaf3", "127.0.0.1", 5003)

        topo.connect("Hub", "Leaf1")
        topo.connect("Hub", "Leaf2")
        topo.connect("Hub", "Leaf3")

        table = build_routing_table(topo, "Hub")

        self.assertEqual(len(table), 3)
        self.assertEqual(table.get_route("Leaf1"), ("127.0.0.1", 5001))
        self.assertEqual(table.get_route("Leaf2"), ("127.0.0.1", 5002))
        self.assertEqual(table.get_route("Leaf3"), ("127.0.0.1", 5003))
        self.assertFalse(table.has_route("Hub"))


class TestRoutePlannerIsolatedAndDisconnected(unittest.TestCase):
    """Tests for isolated sources and unreachable destinations."""

    def test_source_without_links_returns_empty_table(self):
        """Verify an isolated source with no connections produces an empty routing table."""
        topo = NetworkTopology()
        topo.register_node("isolated", "127.0.0.1", 5001)
        topo.register_node("other-1", "127.0.0.1", 5002)
        topo.register_node("other-2", "127.0.0.1", 5003)
        topo.connect("other-1", "other-2")

        table = build_routing_table(topo, "isolated")

        self.assertEqual(len(table), 0)
        self.assertFalse(table.has_route("isolated"))
        self.assertFalse(table.has_route("other-1"))
        self.assertFalse(table.has_route("other-2"))

    def test_disconnected_destinations_are_omitted(self):
        """
        Topology has two disjoint components:
        Component 1: A - B
        Component 2: C - D
        Table for A must only contain B, omitting C and D.
        """
        topo = NetworkTopology()
        topo.register_node("A", "10.0.0.1", 5001)
        topo.register_node("B", "10.0.0.2", 5002)
        topo.register_node("C", "10.0.0.3", 5003)
        topo.register_node("D", "10.0.0.4", 5004)

        topo.connect("A", "B")
        topo.connect("C", "D")

        table = build_routing_table(topo, "A")

        self.assertEqual(len(table), 1)
        self.assertTrue(table.has_route("B"))
        self.assertEqual(table.get_route("B"), ("10.0.0.2", 5002))
        self.assertFalse(table.has_route("C"))
        self.assertFalse(table.has_route("D"))
        self.assertIsNone(table.get_route("C"))
        self.assertIsNone(table.get_route("D"))


class TestRoutePlannerFirstHopEndpointsVerification(unittest.TestCase):
    """Tests proving that multi-hop entries never expose intermediate/final endpoints."""

    def test_all_distant_destinations_use_first_hop_only(self):
        """
        Topology: S -> Hop1(5010) -> Hop2(5020) -> Hop3(5030) -> Target(5040)
        All forwarding entries from S must use (127.0.0.1, 5010).
        """
        topo = NetworkTopology()
        topo.register_node("S", "127.0.0.1", 5000)
        topo.register_node("Hop1", "127.0.0.1", 5010)
        topo.register_node("Hop2", "127.0.0.1", 5020)
        topo.register_node("Hop3", "127.0.0.1", 5030)
        topo.register_node("Target", "127.0.0.1", 5040)

        topo.connect("S", "Hop1")
        topo.connect("Hop1", "Hop2")
        topo.connect("Hop2", "Hop3")
        topo.connect("Hop3", "Target")

        table = build_routing_table(topo, "S")

        first_hop_endpoint = ("127.0.0.1", 5010)
        self.assertEqual(table.get_route("Hop1"), first_hop_endpoint)
        self.assertEqual(table.get_route("Hop2"), first_hop_endpoint)
        self.assertEqual(table.get_route("Hop3"), first_hop_endpoint)
        self.assertEqual(table.get_route("Target"), first_hop_endpoint)

        # Confirm non-first-hop ports are nowhere in the routing table
        for dest, (host, port) in table.get_all_routes().items():
            self.assertEqual(
                port,
                5010,
                f"Route for {dest} has port {port}, expected first hop port 5010",
            )


class TestRoutePlannerIsolationAndUnchangedTopology(unittest.TestCase):
    """Tests verifying state immutability and isolation."""

    def test_topology_state_is_not_modified(self):
        """Verify building a routing table leaves topology nodes, links, and endpoints unchanged."""
        topo = NetworkTopology()
        topo.register_node("A", "127.0.0.1", 5001)
        topo.register_node("B", "127.0.0.1", 5002)
        topo.register_node("C", "127.0.0.1", 5003)
        topo.connect("A", "B")
        topo.connect("B", "C")

        initial_nodes = topo.nodes()
        initial_link_count = topo.link_count()
        initial_endpoints = topo.get_all_endpoints()

        _ = build_routing_table(topo, "A")

        self.assertEqual(topo.nodes(), initial_nodes)
        self.assertEqual(topo.link_count(), initial_link_count)
        self.assertEqual(topo.get_all_endpoints(), initial_endpoints)

    def test_returned_table_mutation_does_not_affect_future_builds(self):
        """Verify modifying the returned RoutingTable does not alter topology or subsequent tables."""
        topo = NetworkTopology()
        topo.register_node("A", "127.0.0.1", 5001)
        topo.register_node("B", "127.0.0.1", 5002)
        topo.connect("A", "B")

        table1 = build_routing_table(topo, "A")
        table1.add_route("CORRUPT", "9.9.9.9", 9999)
        table1.remove_route("B")

        table2 = build_routing_table(topo, "A")
        self.assertTrue(table2.has_route("B"))
        self.assertFalse(table2.has_route("CORRUPT"))
        self.assertEqual(table2.get_route("B"), ("127.0.0.1", 5002))


class TestRoutePlannerDynamicRerouting(unittest.TestCase):
    """Tests verifying table rebuilding after topology changes."""

    def test_rebuilding_table_after_link_removal(self):
        """
        Initial:
          Direct link S-D (5004)
          Alternate path S-R(5002)-D
        Table from S: D -> (127.0.0.1, 5004)

        Disconnect S-D:
        Table from S: D -> (127.0.0.1, 5002) [R's endpoint]

        Reconnect S-D:
        Table from S: D -> (127.0.0.1, 5004)
        """
        topo = NetworkTopology()
        topo.register_node("S", "127.0.0.1", 5001)
        topo.register_node("R", "127.0.0.1", 5002)
        topo.register_node("D", "127.0.0.1", 5004)

        topo.connect("S", "D")
        topo.connect("S", "R")
        topo.connect("R", "D")

        # Initial build: direct link is 1 hop
        table_initial = build_routing_table(topo, "S")
        self.assertEqual(table_initial.get_route("D"), ("127.0.0.1", 5004))
        self.assertEqual(table_initial.get_route("R"), ("127.0.0.1", 5002))

        # Sever direct link
        topo.disconnect("S", "D")
        table_rerouted = build_routing_table(topo, "S")
        self.assertEqual(table_rerouted.get_route("D"), ("127.0.0.1", 5002))
        self.assertEqual(table_rerouted.get_route("R"), ("127.0.0.1", 5002))

        # Restore direct link
        topo.connect("S", "D")
        table_restored = build_routing_table(topo, "S")
        self.assertEqual(table_restored.get_route("D"), ("127.0.0.1", 5004))


class TestWeightedRoutePlannerValidation(unittest.TestCase):
    """Tests for argument validation in build_weighted_routing_table."""

    def setUp(self):
        self.topo = NetworkTopology()
        self.topo.register_node("R1", "127.0.0.1", 5001)

    def test_invalid_topology_type_raises(self):
        """Verify passing a non-NetworkTopology object raises RoutePlannerError."""
        with self.assertRaises(RoutePlannerError):
            build_weighted_routing_table("not-a-topology", "R1")  # type: ignore
        with self.assertRaises(RoutePlannerError):
            build_weighted_routing_table(None, "R1")  # type: ignore
        with self.assertRaises(RoutePlannerError):
            build_weighted_routing_table({}, "R1")  # type: ignore

    def test_unregistered_source_raises_node_not_found(self):
        """Verify unregistered source node raises NodeNotFoundError."""
        with self.assertRaises(NodeNotFoundError):
            build_weighted_routing_table(self.topo, "ghost-router")

    def test_invalid_source_identifiers_raise_invalid_node_error(self):
        """Verify invalid source node identifiers raise InvalidNodeError."""
        invalid_sources = [
            None,
            123,
            True,
            False,
            "",
            "   ",
            " R1",
            "R1 ",
            "\tR1",
            "R1\n",
        ]
        for bad_source in invalid_sources:
            with self.subTest(bad_source=bad_source):
                with self.assertRaises(InvalidNodeError):
                    build_weighted_routing_table(self.topo, bad_source)  # type: ignore


class TestWeightedRoutePlannerExecution(unittest.TestCase):
    """Tests for build_weighted_routing_table execution and route derivation."""

    def test_weighted_route_more_hops_than_min_hop_route(self):
        """
        Verify weighted routing table selects a lower-cost path even when it has more hops.

        Topology:
          S(5001) ---[cost 100]--- D(5004)
          S(5001) ---[cost 2]--- R1(5002) ---[cost 2]--- R2(5003) ---[cost 2]--- D(5004)

        Unweighted BFS route from S:
        - D -> (127.0.0.1, 5004) [direct 1 hop link]

        Weighted route from S:
        - D  -> (127.0.0.1, 5002) [via R1, cost 6 vs 100]
        - R1 -> (127.0.0.1, 5002) [cost 2]
        - R2 -> (127.0.0.1, 5002) [via R1, cost 4]
        """
        topo = NetworkTopology()
        topo.register_node("S", "127.0.0.1", 5001)
        topo.register_node("R1", "127.0.0.1", 5002)
        topo.register_node("R2", "127.0.0.1", 5003)
        topo.register_node("D", "127.0.0.1", 5004)

        topo.connect("S", "D", weight=100)
        topo.connect("S", "R1", weight=2)
        topo.connect("R1", "R2", weight=2)
        topo.connect("R2", "D", weight=2)

        # Baseline unweighted BFS verification
        unweighted_table = build_routing_table(topo, "S")
        self.assertEqual(unweighted_table.get_route("D"), ("127.0.0.1", 5004))

        # Weighted routing table verification
        weighted_table = build_weighted_routing_table(topo, "S")
        self.assertIsInstance(weighted_table, RoutingTable)
        self.assertEqual(len(weighted_table), 3)

        first_hop_endpoint = ("127.0.0.1", 5002)
        self.assertEqual(weighted_table.get_route("R1"), first_hop_endpoint)
        self.assertEqual(weighted_table.get_route("R2"), first_hop_endpoint)
        self.assertEqual(weighted_table.get_route("D"), first_hop_endpoint)

        # Verify source itself is not added
        self.assertFalse(weighted_table.has_route("S"))
        self.assertIsNone(weighted_table.get_route("S"))

    def test_multiple_destinations_sharing_same_first_hop(self):
        """
        Verify branching paths sharing a first-hop all inherit that same first hop endpoint.

        Topology:
          S(5000) ---[1]--- H1(5010) ---[1]--- B1(5020) ---[1]--- L1(5021)
                                      ---[1]--- B2(5030) ---[1]--- L2(5031)
        """
        topo = NetworkTopology()
        topo.register_node("S", "127.0.0.1", 5000)
        topo.register_node("H1", "127.0.0.1", 5010)
        topo.register_node("B1", "127.0.0.1", 5020)
        topo.register_node("B2", "127.0.0.1", 5030)
        topo.register_node("L1", "127.0.0.1", 5021)
        topo.register_node("L2", "127.0.0.1", 5031)

        topo.connect("S", "H1", weight=1)
        topo.connect("H1", "B1", weight=1)
        topo.connect("H1", "B2", weight=1)
        topo.connect("B1", "L1", weight=1)
        topo.connect("B2", "L2", weight=1)

        table = build_weighted_routing_table(topo, "S")
        self.assertEqual(len(table), 5)

        h1_endpoint = ("127.0.0.1", 5010)
        for dest in ["H1", "B1", "B2", "L1", "L2"]:
            self.assertEqual(
                table.get_route(dest),
                h1_endpoint,
                f"Destination {dest} must route via first hop H1",
            )

    def test_direct_neighbor_destinations_match_own_endpoints(self):
        """
        Star topology: Hub directly connected to L1, L2, L3 with distinct weights.
        Every destination's forwarding entry must point directly to its own endpoint.
        """
        topo = NetworkTopology()
        topo.register_node("Hub", "127.0.0.1", 5000)
        topo.register_node("L1", "127.0.0.1", 5001)
        topo.register_node("L2", "127.0.0.1", 5002)
        topo.register_node("L3", "127.0.0.1", 5003)

        topo.connect("Hub", "L1", weight=1)
        topo.connect("Hub", "L2", weight=5)
        topo.connect("Hub", "L3", weight=10)

        table = build_weighted_routing_table(topo, "Hub")
        self.assertEqual(len(table), 3)
        self.assertEqual(table.get_route("L1"), ("127.0.0.1", 5001))
        self.assertEqual(table.get_route("L2"), ("127.0.0.1", 5002))
        self.assertEqual(table.get_route("L3"), ("127.0.0.1", 5003))
        self.assertFalse(table.has_route("Hub"))

    def test_unreachable_destinations_receive_no_route(self):
        """Verify disconnected components or isolated nodes receive no route entries."""
        topo = NetworkTopology()
        topo.register_node("A", "10.0.0.1", 5001)
        topo.register_node("B", "10.0.0.2", 5002)
        topo.register_node("C", "10.0.0.3", 5003)
        topo.register_node("D", "10.0.0.4", 5004)
        topo.register_node("Iso", "10.0.0.5", 5005)

        topo.connect("A", "B", weight=3)
        topo.connect("C", "D", weight=4)

        table_a = build_weighted_routing_table(topo, "A")
        self.assertEqual(len(table_a), 1)
        self.assertEqual(table_a.get_route("B"), ("10.0.0.2", 5002))
        self.assertFalse(table_a.has_route("C"))
        self.assertFalse(table_a.has_route("D"))
        self.assertFalse(table_a.has_route("Iso"))

        table_iso = build_weighted_routing_table(topo, "Iso")
        self.assertEqual(len(table_iso), 0)

    def test_deterministic_results_for_equal_cost_alternatives(self):
        """
        Verify equal-cost paths break ties deterministically based on discovery order.

        Topology:
          S connects to A (cost 5) and B (cost 5). Both connect to D (cost 5).
        """
        topo1 = NetworkTopology()
        topo1.register_node("S", "127.0.0.1", 5001)
        topo1.register_node("A", "127.0.0.1", 5002)
        topo1.register_node("B", "127.0.0.1", 5003)
        topo1.register_node("D", "127.0.0.1", 5004)

        topo1.connect("S", "A", weight=5)  # A inserted first
        topo1.connect("S", "B", weight=5)
        topo1.connect("A", "D", weight=5)
        topo1.connect("B", "D", weight=5)

        table1 = build_weighted_routing_table(topo1, "S")
        self.assertEqual(table1.get_route("D"), ("127.0.0.1", 5002))

        # Topology 2: B inserted first
        topo2 = NetworkTopology()
        topo2.register_node("S", "127.0.0.1", 5001)
        topo2.register_node("A", "127.0.0.1", 5002)
        topo2.register_node("B", "127.0.0.1", 5003)
        topo2.register_node("D", "127.0.0.1", 5004)

        topo2.connect("S", "B", weight=5)  # B inserted first
        topo2.connect("S", "A", weight=5)
        topo2.connect("B", "D", weight=5)
        topo2.connect("A", "D", weight=5)

        table2 = build_weighted_routing_table(topo2, "S")
        self.assertEqual(table2.get_route("D"), ("127.0.0.1", 5003))

    def test_zero_weight_edges_and_cycles(self):
        """Verify zero-weight edges and cycles compute correctly without infinite looping."""
        topo = NetworkTopology()
        topo.register_node("S", "127.0.0.1", 5000)
        topo.register_node("A", "127.0.0.1", 5001)
        topo.register_node("B", "127.0.0.1", 5002)
        topo.register_node("C", "127.0.0.1", 5003)

        topo.connect("S", "A", weight=0)
        topo.connect("A", "B", weight=0)
        topo.connect("B", "S", weight=5)  # Cycle
        topo.connect("B", "C", weight=1.5)

        table = build_weighted_routing_table(topo, "S")
        self.assertEqual(len(table), 3)

        a_endpoint = ("127.0.0.1", 5001)
        self.assertEqual(table.get_route("A"), a_endpoint)
        self.assertEqual(table.get_route("B"), a_endpoint)
        self.assertEqual(table.get_route("C"), a_endpoint)

    def test_weighted_planning_does_not_mutate_topology(self):
        """Verify build_weighted_routing_table leaves the topology completely unaltered."""
        topo = NetworkTopology()
        topo.register_node("A", "127.0.0.1", 5001)
        topo.register_node("B", "127.0.0.1", 5002)
        topo.register_node("C", "127.0.0.1", 5003)
        topo.connect("A", "B", weight=3)
        topo.connect("B", "C", weight=4)

        initial_nodes = topo.nodes()
        initial_link_count = topo.link_count()
        initial_endpoints = topo.get_all_endpoints()
        initial_ab_weight = topo.get_link_weight("A", "B")
        initial_bc_weight = topo.get_link_weight("B", "C")

        _ = build_weighted_routing_table(topo, "A")

        self.assertEqual(topo.nodes(), initial_nodes)
        self.assertEqual(topo.link_count(), initial_link_count)
        self.assertEqual(topo.get_all_endpoints(), initial_endpoints)
        self.assertEqual(topo.get_link_weight("A", "B"), initial_ab_weight)
        self.assertEqual(topo.get_link_weight("B", "C"), initial_bc_weight)

    def test_returned_table_mutation_does_not_affect_future_builds(self):
        """Verify mutating returned table does not affect subsequent builds."""
        topo = NetworkTopology()
        topo.register_node("A", "127.0.0.1", 5001)
        topo.register_node("B", "127.0.0.1", 5002)
        topo.connect("A", "B", weight=2)

        table1 = build_weighted_routing_table(topo, "A")
        table1.add_route("CORRUPT", "9.9.9.9", 9999)
        table1.remove_route("B")

        table2 = build_weighted_routing_table(topo, "A")
        self.assertTrue(table2.has_route("B"))
        self.assertFalse(table2.has_route("CORRUPT"))
        self.assertEqual(table2.get_route("B"), ("127.0.0.1", 5002))

    def test_dynamic_rebuilding_after_weight_or_link_change(self):
        """Verify table updates dynamically when link weights change or links fail."""
        topo = NetworkTopology()
        topo.register_node("S", "127.0.0.1", 5001)
        topo.register_node("R", "127.0.0.1", 5002)
        topo.register_node("D", "127.0.0.1", 5004)

        topo.connect("S", "D", weight=10)
        topo.connect("S", "R", weight=2)
        topo.connect("R", "D", weight=2)

        # Initially, S -> R -> D cost 4 is cheaper than S -> D cost 10
        table1 = build_weighted_routing_table(topo, "S")
        self.assertEqual(table1.get_route("D"), ("127.0.0.1", 5002))

        # Increase R -> D cost to 20; now direct S -> D cost 10 is cheaper
        topo.set_link_weight("R", "D", 20)
        table2 = build_weighted_routing_table(topo, "S")
        self.assertEqual(table2.get_route("D"), ("127.0.0.1", 5004))
        self.assertEqual(table2.get_route("R"), ("127.0.0.1", 5002))


if __name__ == "__main__":
    unittest.main()
