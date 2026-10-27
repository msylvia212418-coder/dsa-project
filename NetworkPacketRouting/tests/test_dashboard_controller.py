"""
test_dashboard_controller.py - Unit tests for the dashboard controller and simulation engine.
"""

from __future__ import annotations

import unittest

from NetworkPacketRouting.dashboard_controller import (
    RouteComparisonResult,
    SimulationResult,
    compare_routes,
    compute_path_cost,
    create_demo_topology,
    generate_graphviz_dot,
    get_topology_metrics,
    simulate_packet_forwarding,
)
from NetworkPacketRouting.topology import (
    NetworkTopology,
    NodeNotFoundError,
)


class TestDashboardController(unittest.TestCase):
    """Tests for dashboard controller logic, route comparisons, and simulation."""

    def setUp(self):
        self.topo = create_demo_topology()

    def test_demo_topology_structure_and_metrics(self):
        """Verify the demo topology builds correctly with expected node and link counts."""
        metrics = get_topology_metrics(self.topo)
        self.assertEqual(metrics["node_count"], 6)
        self.assertEqual(metrics["router_count"], 4)
        self.assertEqual(metrics["receiver_count"], 2)
        self.assertEqual(metrics["link_count"], 7)
        self.assertIn("R1", metrics["routers"])
        self.assertIn("Dest-Alpha", metrics["receivers"])

        # Check expensive direct link vs cheap multi-hop weights
        self.assertEqual(self.topo.get_link_weight("R1", "R2"), 100)
        self.assertEqual(self.topo.get_link_weight("R1", "R3"), 2)
        self.assertEqual(self.topo.get_link_weight("R3", "R2"), 2)
        self.assertEqual(self.topo.get_link_weight("R1", "R4"), 5)
        self.assertEqual(self.topo.get_link_weight("R4", "R2"), 5)

    def test_compare_routes_dijkstra_vs_bfs_divergence(self):
        """
        Verify that R1 -> Dest-Alpha demonstrates a distinct trade-off:
        BFS selects the 2-hop route with cost 101,
        Dijkstra selects the 3-hop route with cost 5.
        """
        res = compare_routes(self.topo, "R1", "Dest-Alpha")

        self.assertIsInstance(res, RouteComparisonResult)
        # BFS (fewest hops)
        self.assertEqual(res.bfs_path, ["R1", "R2", "Dest-Alpha"])
        self.assertEqual(res.bfs_hops, 2)
        self.assertEqual(res.bfs_cost, 101)
        self.assertEqual(res.bfs_next_hop, "R2")

        # Dijkstra (least cost)
        self.assertEqual(res.dijkstra_path, ["R1", "R3", "R2", "Dest-Alpha"])
        self.assertEqual(res.dijkstra_hops, 3)
        self.assertEqual(res.dijkstra_cost, 5)
        self.assertEqual(res.dijkstra_next_hop, "R3")

        # Flags
        self.assertTrue(res.is_different_path)
        self.assertTrue(res.is_cost_divergence)

    def test_compare_routes_convergence(self):
        """Verify that R1 -> Dest-Beta produces identical 2-hop cost-7 paths for both algorithms."""
        res = compare_routes(self.topo, "R1", "Dest-Beta")

        self.assertEqual(res.bfs_path, ["R1", "R4", "Dest-Beta"])
        self.assertEqual(res.dijkstra_path, ["R1", "R4", "Dest-Beta"])
        self.assertEqual(res.bfs_cost, 7)
        self.assertEqual(res.dijkstra_cost, 7)
        self.assertFalse(res.is_different_path)
        self.assertFalse(res.is_cost_divergence)

    def test_compare_routes_unknown_node_raises(self):
        """Verify compare_routes raises NodeNotFoundError for unregistered nodes."""
        with self.assertRaises(NodeNotFoundError):
            compare_routes(self.topo, "UNKNOWN", "Dest-Alpha")
        with self.assertRaises(NodeNotFoundError):
            compare_routes(self.topo, "R1", "UNKNOWN")

    def test_compute_path_cost_zero_or_single_node(self):
        """Verify path cost computation handles empty and single-node paths."""
        self.assertEqual(compute_path_cost(self.topo, []), 0)
        self.assertEqual(compute_path_cost(self.topo, ["R1"]), 0)
        self.assertEqual(compute_path_cost(self.topo, ["R1", "R3", "R2"]), 4)

    def test_simulate_packet_forwarding_weighted_mode(self):
        """Verify simulated packet forwarding over weighted Dijkstra path with TTL tracking."""
        res = simulate_packet_forwarding(
            self.topo,
            source="R1",
            destination="Dest-Alpha",
            payload="Test Payload",
            initial_ttl=8,
            routing_mode="weighted",
        )

        self.assertIsInstance(res, SimulationResult)
        self.assertTrue(res.delivered)
        self.assertEqual(res.status, "DELIVERED")
        self.assertEqual(res.hops_traversed, ["R1", "R3", "R2", "Dest-Alpha"])
        # TTL decrement: 8 -> 7 (R1), 7 -> 6 (R3), 6 -> 5 (R2), reaches Dest-Alpha
        self.assertEqual(res.final_ttl, 5)
        self.assertEqual(len(res.steps), 4)

    def test_simulate_packet_forwarding_unweighted_mode(self):
        """Verify simulated packet forwarding over unweighted BFS path."""
        res = simulate_packet_forwarding(
            self.topo,
            source="R1",
            destination="Dest-Alpha",
            payload="Test BFS",
            initial_ttl=8,
            routing_mode="unweighted",
        )

        self.assertTrue(res.delivered)
        self.assertEqual(res.status, "DELIVERED")
        self.assertEqual(res.hops_traversed, ["R1", "R2", "Dest-Alpha"])
        # TTL decrement: 8 -> 7 (R1), 7 -> 6 (R2), reaches Dest-Alpha
        self.assertEqual(res.final_ttl, 6)

    def test_simulate_packet_forwarding_ttl_expiry(self):
        """Verify that insufficient initial TTL causes packet to drop with DROPPED_TTL."""
        # Dijkstra path has 3 router hops: R1 -> R3 -> R2 -> Dest-Alpha.
        # Initial TTL=2:
        # At R1: TTL=2 > 1 -> forwards to R3, TTL becomes 1.
        # At R3: TTL=1 <= 1 -> DROPPED_TTL!
        res = simulate_packet_forwarding(
            self.topo,
            source="R1",
            destination="Dest-Alpha",
            initial_ttl=2,
            routing_mode="weighted",
        )

        self.assertFalse(res.delivered)
        self.assertEqual(res.status, "DROPPED_TTL")
        self.assertEqual(res.hops_traversed, ["R1", "R3"])
        self.assertEqual(res.final_ttl, 0)

    def test_simulate_packet_forwarding_immediate_delivery(self):
        """Verify source == destination results in immediate delivery without decrement."""
        res = simulate_packet_forwarding(
            self.topo,
            source="Dest-Alpha",
            destination="Dest-Alpha",
            initial_ttl=5,
        )
        self.assertTrue(res.delivered)
        self.assertEqual(res.status, "DELIVERED")
        self.assertEqual(res.final_ttl, 5)
        self.assertEqual(res.hops_traversed, ["Dest-Alpha"])

    def test_simulate_packet_forwarding_unreachable_node(self):
        """Verify simulated packet to an isolated/disconnected node results in DROPPED_NO_ROUTE."""
        self.topo.register_node("Isolated-Dest", "10.0.0.1", 9999)

        res = simulate_packet_forwarding(
            self.topo,
            source="R1",
            destination="Isolated-Dest",
            initial_ttl=8,
        )

        self.assertFalse(res.delivered)
        self.assertEqual(res.status, "NO_ROUTE")
        self.assertIn("No route", res.summary)

    def test_dynamic_link_failure_and_route_recalculation(self):
        """Verify dynamic disconnection of links triggers correct rerouting."""
        # Sever the primary cheap multi-hop link R1 - R3
        self.topo.disconnect("R1", "R3")

        # Now Dijkstra chooses alternative multi-hop route via R4 (cost 5 + 5 + 1 = 11),
        # which is still much cheaper than direct R1 -> R2 (cost 101).
        res1 = compare_routes(self.topo, "R1", "Dest-Alpha")
        self.assertEqual(res1.dijkstra_path, ["R1", "R4", "R2", "Dest-Alpha"])
        self.assertEqual(res1.dijkstra_cost, 11)

        # Sever the second multi-hop link R1 - R4
        self.topo.disconnect("R1", "R4")

        # Now Dijkstra must fall back to the direct expensive link R1 -> R2 (cost 101)
        res2 = compare_routes(self.topo, "R1", "Dest-Alpha")
        self.assertEqual(res2.dijkstra_path, ["R1", "R2", "Dest-Alpha"])
        self.assertEqual(res2.dijkstra_cost, 101)

    def test_generate_graphviz_dot_syntax(self):
        """Verify Graphviz DOT script contains expected node styles, highlighted edges, and severed links."""
        dot = generate_graphviz_dot(
            self.topo,
            highlight_path=["R1", "R3", "R2", "Dest-Alpha"],
            failed_links={("R1", "R2")},
        )

        self.assertTrue(dot.startswith("graph NetworkTopology {"))
        self.assertTrue(dot.endswith("}"))
        self.assertIn('"R1" [', dot)
        self.assertIn('"Dest-Alpha" [', dot)
        # Highlighted edge styling
        self.assertIn('#F59E0B', dot)
        # Failed/severed edge styling
        self.assertIn('SEVERED', dot)


if __name__ == "__main__":
    unittest.main()
