"""
test_topology.py - Comprehensive unit tests for NetworkTopology.

Verifies:
- Node and endpoint registration, validation, and collision prevention.
- Link creation, removal, self-loop rejection, and neighbor inspection.
- Shortest-hop path discovery (BFS) including edge cases and equal-length paths.
- Deterministic tie-breaking across equal-length routes based on insertion order.
- Dynamic route recalculation upon link additions/removals.
- Next-hop calculation for direct, multi-hop, self, and disconnected nodes.
- Defensive copy isolation for endpoints, node lists, and the underlying graph.
- Exception handling for invalid inputs and unregistered nodes.
"""

from __future__ import annotations

import unittest

from NetworkPacketRouting.graph import InvalidNodeError, NetworkGraph
from NetworkPacketRouting.topology import (
    DuplicateEndpointError,
    DuplicateNodeError,
    InvalidEndpointError,
    LinkError,
    NetworkTopology,
    NodeNotFoundError,
    TopologyError,
)


class TestTopologyRegistration(unittest.TestCase):
    """Tests for node and endpoint registration, validation, and querying."""

    def test_register_single_node(self):
        """Verify registering a valid node and endpoint succeeds."""
        topo = NetworkTopology()
        topo.register_node("router-1", "127.0.0.1", 5001)

        self.assertEqual(topo.node_count(), 1)
        self.assertTrue(topo.has_node("router-1"))
        self.assertEqual(topo.get_endpoint("router-1"), ("127.0.0.1", 5001))

    def test_register_multiple_nodes(self):
        """Verify registering multiple distinct nodes and endpoints."""
        topo = NetworkTopology()
        topo.register_node("R1", "10.0.0.1", 5001)
        topo.register_node("R2", "10.0.0.2", 5002)
        topo.register_node("R3", "10.0.0.3", 5003)

        self.assertEqual(topo.node_count(), 3)
        self.assertEqual(topo.nodes(), ["R1", "R2", "R3"])
        self.assertEqual(topo.get_endpoint("R1"), ("10.0.0.1", 5001))
        self.assertEqual(topo.get_endpoint("R2"), ("10.0.0.2", 5002))
        self.assertEqual(topo.get_endpoint("R3"), ("10.0.0.3", 5003))

    def test_register_duplicate_node_raises_duplicate_node_error(self):
        """Verify registering an already-registered node without overwrite raises DuplicateNodeError."""
        topo = NetworkTopology()
        topo.register_node("R1", "127.0.0.1", 5001)

        with self.assertRaises(DuplicateNodeError):
            topo.register_node("R1", "127.0.0.1", 5002)

        # Confirm original endpoint was preserved
        self.assertEqual(topo.get_endpoint("R1"), ("127.0.0.1", 5001))

    def test_register_duplicate_node_with_overwrite_succeeds(self):
        """Verify registering an existing node with overwrite=True updates the endpoint."""
        topo = NetworkTopology()
        topo.register_node("R1", "127.0.0.1", 5001)
        topo.register_node("R1", "127.0.0.1", 5002, overwrite=True)

        self.assertEqual(topo.node_count(), 1)
        self.assertEqual(topo.get_endpoint("R1"), ("127.0.0.1", 5002))

    def test_register_duplicate_endpoint_different_node_raises(self):
        """Verify two distinct nodes cannot bind to the exact same (host, port)."""
        topo = NetworkTopology()
        topo.register_node("R1", "127.0.0.1", 5001)

        with self.assertRaises(DuplicateEndpointError):
            topo.register_node("R2", "127.0.0.1", 5001)

        self.assertFalse(topo.has_node("R2"))

    def test_unregister_existing_node(self):
        """Verify unregistering a node removes it and its endpoint."""
        topo = NetworkTopology()
        topo.register_node("R1", "127.0.0.1", 5001)

        self.assertTrue(topo.unregister_node("R1"))
        self.assertEqual(topo.node_count(), 0)
        self.assertFalse(topo.has_node("R1"))
        with self.assertRaises(NodeNotFoundError):
            topo.get_endpoint("R1")

    def test_unregister_nonexistent_node_returns_false(self):
        """Verify unregistering an unknown node safely returns False."""
        topo = NetworkTopology()
        self.assertFalse(topo.unregister_node("ghost"))

    def test_has_node_queries(self):
        """Verify has_node returns correct boolean without error."""
        topo = NetworkTopology()
        topo.register_node("A", "127.0.0.1", 5001)
        self.assertTrue(topo.has_node("A"))
        self.assertFalse(topo.has_node("B"))

    def test_get_all_endpoints(self):
        """Verify get_all_endpoints returns a complete mapping."""
        topo = NetworkTopology()
        topo.register_node("A", "127.0.0.1", 5001)
        topo.register_node("B", "127.0.0.1", 5002)

        endpoints = topo.get_all_endpoints()
        self.assertEqual(
            endpoints,
            {"A": ("127.0.0.1", 5001), "B": ("127.0.0.1", 5002)},
        )

    def test_invalid_node_identifiers(self):
        """Verify invalid node identifiers are rejected with InvalidNodeError."""
        topo = NetworkTopology()
        invalid_nodes = [
            None,
            123,
            True,
            False,
            [],
            "",
            "   ",
            " R1",
            "R1 ",
            "\tR1",
            "R1\n",
        ]
        for bad_id in invalid_nodes:
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(InvalidNodeError):
                    topo.register_node(bad_id, "127.0.0.1", 5001)
                with self.assertRaises(InvalidNodeError):
                    topo.has_node(bad_id)
                with self.assertRaises(InvalidNodeError):
                    topo.unregister_node(bad_id)
                with self.assertRaises(InvalidNodeError):
                    topo.get_endpoint(bad_id)

    def test_invalid_host_values(self):
        """Verify invalid host values are rejected with InvalidEndpointError."""
        topo = NetworkTopology()
        invalid_hosts = [
            None,
            127,
            True,
            False,
            "",
            "   ",
            " 127.0.0.1",
            "127.0.0.1 ",
            "\tlocalhost",
        ]
        for bad_host in invalid_hosts:
            with self.subTest(bad_host=bad_host):
                with self.assertRaises(InvalidEndpointError):
                    topo.register_node("R1", bad_host, 5001)

    def test_invalid_port_values(self):
        """Verify invalid port values are rejected with InvalidEndpointError."""
        topo = NetworkTopology()
        invalid_ports = [
            None,
            "5001",
            5001.0,
            True,  # booleans inherit from int in Python
            False,
            0,
            -1,
            -5000,
            65536,
            70000,
        ]
        for bad_port in invalid_ports:
            with self.subTest(bad_port=bad_port):
                with self.assertRaises(InvalidEndpointError):
                    topo.register_node("R1", "127.0.0.1", bad_port)

    def test_port_boundaries(self):
        """Verify boundary ports 1 and 65535 succeed."""
        topo = NetworkTopology()
        topo.register_node("min-port", "127.0.0.1", 1)
        topo.register_node("max-port", "127.0.0.1", 65535)

        self.assertEqual(topo.get_endpoint("min-port"), ("127.0.0.1", 1))
        self.assertEqual(topo.get_endpoint("max-port"), ("127.0.0.1", 65535))


class TestTopologyLinks(unittest.TestCase):
    """Tests for link creation, removal, and neighbor inspection."""

    def setUp(self):
        self.topo = NetworkTopology()
        self.topo.register_node("A", "10.0.0.1", 5001)
        self.topo.register_node("B", "10.0.0.2", 5002)
        self.topo.register_node("C", "10.0.0.3", 5003)

    def test_connect_two_registered_nodes(self):
        """Verify connecting two nodes creates an undirected link."""
        self.assertTrue(self.topo.connect("A", "B"))
        self.assertTrue(self.topo.has_link("A", "B"))
        self.assertTrue(self.topo.has_link("B", "A"))
        self.assertEqual(self.topo.link_count(), 1)
        self.assertEqual(self.topo.neighbors("A"), ["B"])
        self.assertEqual(self.topo.neighbors("B"), ["A"])

    def test_add_link_alias(self):
        """Verify add_link functions identically to connect."""
        self.assertTrue(self.topo.add_link("A", "C"))
        self.assertTrue(self.topo.has_link("A", "C"))

    def test_duplicate_connect_returns_false(self):
        """Verify adding an existing link returns False without creating a duplicate."""
        self.assertTrue(self.topo.connect("A", "B"))
        self.assertFalse(self.topo.connect("A", "B"))
        self.assertFalse(self.topo.connect("B", "A"))
        self.assertEqual(self.topo.link_count(), 1)
        self.assertEqual(self.topo.neighbors("A"), ["B"])

    def test_disconnect_existing_link(self):
        """Verify disconnecting an existing link removes it symmetrically."""
        self.topo.connect("A", "B")
        self.assertTrue(self.topo.disconnect("A", "B"))
        self.assertFalse(self.topo.has_link("A", "B"))
        self.assertFalse(self.topo.has_link("B", "A"))
        self.assertEqual(self.topo.link_count(), 0)
        self.assertEqual(self.topo.neighbors("A"), [])
        self.assertEqual(self.topo.neighbors("B"), [])

    def test_remove_link_alias(self):
        """Verify remove_link functions identically to disconnect."""
        self.topo.connect("A", "B")
        self.assertTrue(self.topo.remove_link("A", "B"))
        self.assertFalse(self.topo.has_link("A", "B"))

    def test_disconnect_nonexistent_link_returns_false(self):
        """Verify disconnecting a link that does not exist returns False."""
        self.assertFalse(self.topo.disconnect("A", "B"))

    def test_connect_unregistered_node_raises_node_not_found(self):
        """Verify connecting an unregistered node raises NodeNotFoundError."""
        with self.assertRaises(NodeNotFoundError):
            self.topo.connect("A", "ghost")
        with self.assertRaises(NodeNotFoundError):
            self.topo.connect("ghost", "B")

    def test_disconnect_unregistered_node_raises_node_not_found(self):
        """Verify disconnecting with an unregistered node raises NodeNotFoundError."""
        with self.assertRaises(NodeNotFoundError):
            self.topo.disconnect("A", "ghost")

    def test_connect_self_loop_raises_link_error(self):
        """Verify self-loops are rejected with LinkError."""
        with self.assertRaises(LinkError):
            self.topo.connect("A", "A")

    def test_has_link_with_unregistered_node_returns_false(self):
        """Verify has_link returns False if either node is unregistered."""
        self.assertFalse(self.topo.has_link("A", "unregistered"))
        self.assertFalse(self.topo.has_link("unregistered", "A"))

    def test_unregister_node_cleans_up_incident_links(self):
        """Verify unregistering a node also removes all its links."""
        self.topo.connect("A", "B")
        self.topo.connect("A", "C")
        self.assertEqual(self.topo.link_count(), 2)

        self.topo.unregister_node("A")
        self.assertEqual(self.topo.link_count(), 0)
        self.assertEqual(self.topo.neighbors("B"), [])
        self.assertEqual(self.topo.neighbors("C"), [])

    def test_neighbors_unregistered_node_raises(self):
        """Verify neighbors() raises NodeNotFoundError for unregistered node."""
        with self.assertRaises(NodeNotFoundError):
            self.topo.neighbors("ghost")


class TestTopologyShortestPath(unittest.TestCase):
    """Tests for BFS unweighted shortest path computation."""

    def setUp(self):
        self.topo = NetworkTopology()
        for i in range(1, 7):
            self.topo.register_node(f"R{i}", f"10.0.0.{i}", 5000 + i)

    def test_shortest_path_source_equals_destination(self):
        """Verify source == destination returns [source]."""
        path = self.topo.shortest_path("R1", "R1")
        self.assertEqual(path, ["R1"])

    def test_shortest_path_direct_neighbor(self):
        """Verify single hop directly connected nodes."""
        self.topo.connect("R1", "R2")
        self.assertEqual(self.topo.shortest_path("R1", "R2"), ["R1", "R2"])
        self.assertEqual(self.topo.shortest_path("R2", "R1"), ["R2", "R1"])

    def test_shortest_path_two_hops(self):
        """Verify 2-hop path through an intermediate router."""
        self.topo.connect("R1", "R2")
        self.topo.connect("R2", "R3")
        self.assertEqual(
            self.topo.shortest_path("R1", "R3"),
            ["R1", "R2", "R3"],
        )

    def test_shortest_path_linear_chain(self):
        """Verify multi-hop path along a linear chain R1-R2-R3-R4-R5-R6."""
        for i in range(1, 6):
            self.topo.connect(f"R{i}", f"R{i+1}")

        expected = ["R1", "R2", "R3", "R4", "R5", "R6"]
        self.assertEqual(self.topo.shortest_path("R1", "R6"), expected)
        self.assertEqual(
            self.topo.shortest_path("R6", "R1"), list(reversed(expected))
        )

    def test_shortest_path_diamond_topology(self):
        """
        Verify shortest path through diamond topology:
              R1
             /  \\
            R2   R3
             \\  /
              R4
        """
        self.topo.connect("R1", "R2")
        self.topo.connect("R1", "R3")
        self.topo.connect("R2", "R4")
        self.topo.connect("R3", "R4")

        # Because R1-R2 was connected before R1-R3, BFS explores R2 first
        self.assertEqual(
            self.topo.shortest_path("R1", "R4"),
            ["R1", "R2", "R4"],
        )

    def test_shortest_path_unreachable_node_returns_none(self):
        """Verify isolated node returns None."""
        self.topo.connect("R1", "R2")
        # R3 has no links
        self.assertIsNone(self.topo.shortest_path("R1", "R3"))

    def test_shortest_path_disconnected_components_returns_none(self):
        """Verify nodes in separate components return None."""
        self.topo.connect("R1", "R2")
        self.topo.connect("R3", "R4")
        self.assertIsNone(self.topo.shortest_path("R1", "R4"))

    def test_shortest_path_unregistered_source_raises(self):
        """Verify unregistered source raises NodeNotFoundError."""
        with self.assertRaises(NodeNotFoundError):
            self.topo.shortest_path("ghost", "R1")

    def test_shortest_path_unregistered_destination_raises(self):
        """Verify unregistered destination raises NodeNotFoundError."""
        with self.assertRaises(NodeNotFoundError):
            self.topo.shortest_path("R1", "ghost")

    def test_shortest_path_invalid_node_identifiers_raise(self):
        """Verify invalid node identifiers raise InvalidNodeError."""
        with self.assertRaises(InvalidNodeError):
            self.topo.shortest_path("", "R1")
        with self.assertRaises(InvalidNodeError):
            self.topo.shortest_path("R1", None)


class TestTopologyEqualLengthAlternatives(unittest.TestCase):
    """
    Tests proving deterministic tie-breaking when multiple equal-length shortest paths exist.

    Tie-breaking contract:
    BFS explores neighbors in the order they were inserted. When two paths have
    the same minimum hop count, the path passing through the earlier-connected
    neighbor is explored first, reaching the destination first.
    """

    def test_equal_length_paths_follows_insertion_order(self):
        """
        Topology:
              S
             / \\
            A   B
             \\ /
              D

        Case 1: S connected to A first, then B.
        Expected path: [S, A, D]
        """
        topo1 = NetworkTopology()
        for idx, node in enumerate(["S", "A", "B", "D"], start=1):
            topo1.register_node(node, "127.0.0.1", 5000 + idx)

        topo1.connect("S", "A")  # A inserted first
        topo1.connect("S", "B")
        topo1.connect("A", "D")
        topo1.connect("B", "D")

        self.assertEqual(topo1.shortest_path("S", "D"), ["S", "A", "D"])

    def test_equal_length_paths_swapped_insertion_order(self):
        """
        Same topology, but S connected to B first, then A.
        Expected path: [S, B, D]
        """
        topo2 = NetworkTopology()
        for idx, node in enumerate(["S", "A", "B", "D"], start=1):
            topo2.register_node(node, "127.0.0.1", 5000 + idx)

        topo2.connect("S", "B")  # B inserted first
        topo2.connect("S", "A")
        topo2.connect("B", "D")
        topo2.connect("A", "D")

        self.assertEqual(topo2.shortest_path("S", "D"), ["S", "B", "D"])

    def test_three_equal_alternatives(self):
        """
        S connected to X, Y, Z (in that order).
        All three connected to D.
        Expected path: [S, X, D]
        """
        topo = NetworkTopology()
        for idx, node in enumerate(["S", "X", "Y", "Z", "D"], start=1):
            topo.register_node(node, "127.0.0.1", 5000 + idx)

        topo.connect("S", "X")
        topo.connect("S", "Y")
        topo.connect("S", "Z")
        topo.connect("X", "D")
        topo.connect("Y", "D")
        topo.connect("Z", "D")

        self.assertEqual(topo.shortest_path("S", "D"), ["S", "X", "D"])


class TestTopologyDynamicRerouting(unittest.TestCase):
    """Tests verifying path recalculation upon link additions and removals."""

    def test_link_removal_forces_longer_alternative(self):
        """
        Initial:
          Direct link S-D (1 hop)
          Alternative path S-R-D (2 hops)
        Disconnect S-D:
          Reroutes to S-R-D (2 hops)
        Reconnect S-D:
          Restores S-D (1 hop)
        """
        topo = NetworkTopology()
        for idx, node in enumerate(["S", "R", "D"], start=1):
            topo.register_node(node, "127.0.0.1", 5000 + idx)

        topo.connect("S", "D")
        topo.connect("S", "R")
        topo.connect("R", "D")

        # Initial: direct link is 1 hop
        self.assertEqual(topo.shortest_path("S", "D"), ["S", "D"])
        self.assertEqual(topo.next_hop("S", "D"), "D")

        # Sever direct link
        topo.disconnect("S", "D")
        self.assertEqual(topo.shortest_path("S", "D"), ["S", "R", "D"])
        self.assertEqual(topo.next_hop("S", "D"), "R")

        # Restore direct link
        topo.connect("S", "D")
        self.assertEqual(topo.shortest_path("S", "D"), ["S", "D"])
        self.assertEqual(topo.next_hop("S", "D"), "D")

    def test_link_removal_reroutes_between_equal_alternatives(self):
        """
        Initial equal paths: S-A-D and S-B-D. (S-A-D selected).
        Disconnect S-A -> immediately falls back to S-B-D.
        """
        topo = NetworkTopology()
        for idx, node in enumerate(["S", "A", "B", "D"], start=1):
            topo.register_node(node, "127.0.0.1", 5000 + idx)

        topo.connect("S", "A")
        topo.connect("S", "B")
        topo.connect("A", "D")
        topo.connect("B", "D")

        self.assertEqual(topo.shortest_path("S", "D"), ["S", "A", "D"])

        topo.disconnect("S", "A")
        self.assertEqual(topo.shortest_path("S", "D"), ["S", "B", "D"])
        self.assertEqual(topo.next_hop("S", "D"), "B")


class TestTopologyNextHop(unittest.TestCase):
    """Tests for next_hop calculation."""

    def setUp(self):
        self.topo = NetworkTopology()
        self.topo.register_node("A", "10.0.0.1", 5001)
        self.topo.register_node("B", "10.0.0.2", 5002)
        self.topo.register_node("C", "10.0.0.3", 5003)
        self.topo.register_node("D", "10.0.0.4", 5004)

        self.topo.connect("A", "B")
        self.topo.connect("B", "C")
        self.topo.connect("C", "D")

    def test_next_hop_source_equals_dest_returns_none(self):
        """Verify next_hop returns None when source equals destination."""
        self.assertIsNone(self.topo.next_hop("A", "A"))

    def test_next_hop_direct_neighbor(self):
        """Verify next_hop to a direct neighbor returns that neighbor."""
        self.assertEqual(self.topo.next_hop("A", "B"), "B")
        self.assertEqual(self.topo.next_hop("B", "A"), "A")

    def test_next_hop_multi_hop(self):
        """Verify next_hop to distant node returns the first intermediate hop."""
        # A -> B -> C -> D
        self.assertEqual(self.topo.next_hop("A", "C"), "B")
        self.assertEqual(self.topo.next_hop("A", "D"), "B")
        self.assertEqual(self.topo.next_hop("B", "D"), "C")

    def test_next_hop_unreachable_node_returns_none(self):
        """Verify next_hop to an unreachable node returns None."""
        self.topo.register_node("isolated", "10.0.0.99", 5099)
        self.assertIsNone(self.topo.next_hop("A", "isolated"))

    def test_next_hop_unregistered_nodes_raise(self):
        """Verify next_hop with unregistered nodes raises NodeNotFoundError."""
        with self.assertRaises(NodeNotFoundError):
            self.topo.next_hop("A", "ghost")
        with self.assertRaises(NodeNotFoundError):
            self.topo.next_hop("ghost", "A")

    def test_next_hop_invalid_identifiers_raise(self):
        """Verify next_hop with invalid node identifiers raises InvalidNodeError."""
        with self.assertRaises(InvalidNodeError):
            self.topo.next_hop("", "A")
        with self.assertRaises(InvalidNodeError):
            self.topo.next_hop("A", None)


class TestTopologyDefensiveCopies(unittest.TestCase):
    """Tests verifying that internal topology state cannot be corrupted via returned objects."""

    def test_endpoints_dict_mutation_isolation(self):
        """Verify mutating get_all_endpoints() dictionary does not corrupt topology."""
        topo = NetworkTopology()
        topo.register_node("A", "127.0.0.1", 5001)

        endpoints = topo.get_all_endpoints()
        endpoints["A"] = ("999.999.999.999", 9999)
        endpoints["INJECTED"] = ("127.0.0.1", 5002)

        # Topology internal state must be unchanged
        self.assertEqual(topo.get_endpoint("A"), ("127.0.0.1", 5001))
        self.assertFalse(topo.has_node("INJECTED"))
        self.assertEqual(topo.node_count(), 1)

    def test_nodes_list_mutation_isolation(self):
        """Verify mutating nodes() list does not corrupt topology."""
        topo = NetworkTopology()
        topo.register_node("A", "127.0.0.1", 5001)

        nodes_list = topo.nodes()
        nodes_list.append("INJECTED")

        self.assertEqual(topo.nodes(), ["A"])
        self.assertFalse(topo.has_node("INJECTED"))

    def test_neighbors_list_mutation_isolation(self):
        """Verify mutating neighbors() list does not affect topology graph."""
        topo = NetworkTopology()
        topo.register_node("A", "127.0.0.1", 5001)
        topo.register_node("B", "127.0.0.1", 5002)
        topo.connect("A", "B")

        nbrs = topo.neighbors("A")
        nbrs.append("FAKE")

        self.assertEqual(topo.neighbors("A"), ["B"])

    def test_get_graph_preserves_nodes_neighbors_edge_count_and_traversal_order(self):
        """
        Verify the copy returned by get_graph() has identical nodes,
        neighbors (in insertion order), edge count, and BFS/DFS traversal order.
        """
        topo = NetworkTopology()
        for idx, name in enumerate(["R1", "R2", "R3", "R4", "R5"], start=1):
            topo.register_node(name, "127.0.0.1", 5000 + idx)

        topo.connect("R1", "R2")
        topo.connect("R1", "R3")
        topo.connect("R2", "R4")
        topo.connect("R3", "R5")

        graph_copy = topo.get_graph()

        # Same type and independent instance
        self.assertIsInstance(graph_copy, NetworkGraph)

        # Same nodes and counts
        self.assertEqual(graph_copy.node_count(), topo.node_count())
        self.assertEqual(graph_copy.nodes(), topo.nodes())
        self.assertEqual(graph_copy.edge_count(), topo.link_count())

        # Same neighbors in exact insertion order for every node
        for node in topo.nodes():
            self.assertEqual(graph_copy.neighbors(node), topo.neighbors(node))

        # Same traversal order
        self.assertEqual(
            graph_copy.bfs("R1"),
            ["R1", "R2", "R3", "R4", "R5"],
        )
        self.assertEqual(
            graph_copy.dfs("R1"),
            ["R1", "R2", "R4", "R3", "R5"],
        )

    def test_get_graph_mutation_isolation_additions(self):
        """Verify adding nodes and edges to the copy does not modify the topology."""
        topo = NetworkTopology()
        topo.register_node("A", "127.0.0.1", 5001)
        topo.register_node("B", "127.0.0.1", 5002)
        topo.connect("A", "B")

        graph_copy = topo.get_graph()
        graph_copy.add_node("EXTERNAL")
        graph_copy.add_edge("B", "EXTERNAL")

        # Topology internal graph must remain intact
        self.assertFalse(topo.has_node("EXTERNAL"))
        self.assertNotIn("EXTERNAL", topo.nodes())
        self.assertEqual(topo.node_count(), 2)
        self.assertEqual(topo.link_count(), 1)
        self.assertEqual(topo.neighbors("B"), ["A"])

    def test_get_graph_mutation_isolation_deletions(self):
        """Verify removing nodes and edges from the copy does not modify the topology."""
        topo = NetworkTopology()
        topo.register_node("A", "127.0.0.1", 5001)
        topo.register_node("B", "127.0.0.1", 5002)
        topo.connect("A", "B")

        graph_copy = topo.get_graph()
        graph_copy.remove_edge("A", "B")
        graph_copy.remove_node("A")

        # Topology internal graph must remain intact
        self.assertTrue(topo.has_node("A"))
        self.assertTrue(topo.has_link("A", "B"))
        self.assertEqual(topo.node_count(), 2)
        self.assertEqual(topo.link_count(), 1)
        self.assertEqual(topo.shortest_path("A", "B"), ["A", "B"])

    def test_topology_mutation_after_get_graph_does_not_affect_copy(self):
        """Verify mutating the topology after get_graph() does not alter the previous copy."""
        topo = NetworkTopology()
        topo.register_node("A", "127.0.0.1", 5001)
        topo.register_node("B", "127.0.0.1", 5002)
        topo.connect("A", "B")

        graph_copy = topo.get_graph()

        # Mutate topology after taking the copy
        topo.register_node("C", "127.0.0.1", 5003)
        topo.connect("B", "C")
        topo.disconnect("A", "B")

        # The previously obtained copy must remain in its snapshot state
        self.assertFalse(graph_copy.has_node("C"))
        self.assertTrue(graph_copy.has_edge("A", "B"))
        self.assertFalse(graph_copy.has_edge("B", "C"))
        self.assertEqual(graph_copy.node_count(), 2)
        self.assertEqual(graph_copy.edge_count(), 1)


if __name__ == "__main__":
    unittest.main()
