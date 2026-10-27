"""
test_graph.py - Comprehensive tests for NetworkGraph, BFS, and DFS.

Covers:
- Empty graph and single-node graph.
- Adding and removing nodes and edges.
- Undirected adjacency and duplicate edges.
- Invalid node identifiers (non-string, empty, whitespace, leading/trailing spaces).
- Self-loop rejection.
- BFS order on a small graph with multiple paths.
- DFS order on a small graph with multiple paths.
- Cycles and disconnected components.
- Unknown traversal starting nodes.
- BFS/DFS parent mappings (bfs_tree / dfs_tree).
- Deterministic results across repeated calls.
- Deep linear graph to confirm DFS does not depend on recursion.
- Auto-creation of nodes by add_edge.
- Safe removal of nonexistent nodes/edges.
- Neighbor list isolation (returned list cannot mutate internals).

Test topologies and expected visit orders are documented inline.
"""

from __future__ import annotations

import sys
import unittest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))

from NetworkPacketRouting.graph import (
    GraphError,
    InvalidNodeError,
    NetworkGraph,
    NodeNotFoundError,
)


class TestNetworkGraphStructure(unittest.TestCase):
    """Tests for graph structure operations: nodes, edges, neighbors."""

    # ------------------------------------------------------------------
    # Empty and single-node graphs
    # ------------------------------------------------------------------

    def test_empty_graph(self):
        """Verify a new graph has zero nodes and edges."""
        g = NetworkGraph()
        self.assertEqual(g.node_count(), 0)
        self.assertEqual(g.edge_count(), 0)
        self.assertEqual(g.nodes(), [])

    def test_single_node_graph(self):
        """Verify a graph with one node and no edges."""
        g = NetworkGraph()
        self.assertTrue(g.add_node("A"))
        self.assertEqual(g.node_count(), 1)
        self.assertEqual(g.edge_count(), 0)
        self.assertTrue(g.has_node("A"))
        self.assertEqual(g.neighbors("A"), [])

    # ------------------------------------------------------------------
    # Adding nodes
    # ------------------------------------------------------------------

    def test_add_node_returns_true_for_new_node(self):
        """Verify add_node returns True for a newly added node."""
        g = NetworkGraph()
        self.assertTrue(g.add_node("router-1"))

    def test_add_node_returns_false_for_duplicate(self):
        """Verify add_node returns False when node already exists."""
        g = NetworkGraph()
        g.add_node("router-1")
        self.assertFalse(g.add_node("router-1"))
        self.assertEqual(g.node_count(), 1)

    def test_nodes_returns_sorted_list(self):
        """Verify nodes() returns identifiers in sorted order."""
        g = NetworkGraph()
        for name in ["C", "A", "B"]:
            g.add_node(name)
        self.assertEqual(g.nodes(), ["A", "B", "C"])

    # ------------------------------------------------------------------
    # Invalid node identifiers
    # ------------------------------------------------------------------

    def test_invalid_node_non_string_types(self):
        """Verify non-string types are rejected with InvalidNodeError."""
        g = NetworkGraph()
        for bad in [42, 3.14, None, [], {}, True, False]:
            with self.assertRaises(InvalidNodeError, msg=f"Should reject {type(bad).__name__}"):
                g.add_node(bad)

    def test_invalid_node_empty_string(self):
        """Verify empty string is rejected."""
        g = NetworkGraph()
        with self.assertRaises(InvalidNodeError):
            g.add_node("")

    def test_invalid_node_whitespace_only(self):
        """Verify whitespace-only strings are rejected."""
        g = NetworkGraph()
        for ws in [" ", "  ", "\t", "\n", " \t\n "]:
            with self.assertRaises(InvalidNodeError, msg=f"Should reject {ws!r}"):
                g.add_node(ws)

    def test_invalid_node_leading_trailing_whitespace(self):
        """Verify identifiers with leading or trailing whitespace are rejected."""
        g = NetworkGraph()
        for bad in [" A", "A ", " A ", "\tB", "B\n"]:
            with self.assertRaises(InvalidNodeError, msg=f"Should reject {bad!r}"):
                g.add_node(bad)

    # ------------------------------------------------------------------
    # Adding edges
    # ------------------------------------------------------------------

    def test_add_edge_creates_undirected_adjacency(self):
        """Verify add_edge creates symmetric adjacency."""
        g = NetworkGraph()
        g.add_edge("A", "B")
        self.assertIn("B", g.neighbors("A"))
        self.assertIn("A", g.neighbors("B"))
        self.assertEqual(g.edge_count(), 1)

    def test_add_edge_auto_creates_nodes(self):
        """Verify add_edge automatically creates endpoint nodes that don't exist."""
        g = NetworkGraph()
        self.assertEqual(g.node_count(), 0)
        g.add_edge("X", "Y")
        self.assertTrue(g.has_node("X"))
        self.assertTrue(g.has_node("Y"))
        self.assertEqual(g.node_count(), 2)

    def test_add_edge_returns_true_for_new_edge(self):
        """Verify add_edge returns True for a newly created edge."""
        g = NetworkGraph()
        self.assertTrue(g.add_edge("A", "B"))

    def test_duplicate_edge_returns_false(self):
        """Verify duplicate add_edge calls return False and don't create duplicates."""
        g = NetworkGraph()
        g.add_edge("A", "B")
        self.assertFalse(g.add_edge("A", "B"))
        self.assertFalse(g.add_edge("B", "A"))  # Reverse direction is same edge
        self.assertEqual(g.edge_count(), 1)
        self.assertEqual(g.neighbors("A"), ["B"])

    def test_self_loop_rejected(self):
        """Verify self-loops raise GraphError."""
        g = NetworkGraph()
        g.add_node("A")
        with self.assertRaises(GraphError):
            g.add_edge("A", "A")

    def test_add_edge_invalid_identifiers(self):
        """Verify add_edge rejects invalid node identifiers."""
        g = NetworkGraph()
        with self.assertRaises(InvalidNodeError):
            g.add_edge("", "B")
        with self.assertRaises(InvalidNodeError):
            g.add_edge("A", " B")

    # ------------------------------------------------------------------
    # Removing nodes
    # ------------------------------------------------------------------

    def test_remove_existing_node(self):
        """Verify removing an existing node removes it and its edges."""
        g = NetworkGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        self.assertTrue(g.remove_node("A"))
        self.assertFalse(g.has_node("A"))
        self.assertEqual(g.neighbors("B"), [])
        self.assertEqual(g.neighbors("C"), [])
        self.assertEqual(g.edge_count(), 0)

    def test_remove_nonexistent_node_returns_false(self):
        """Verify removing a nonexistent node is safe and returns False."""
        g = NetworkGraph()
        self.assertFalse(g.remove_node("ghost"))

    def test_remove_node_invalid_identifier(self):
        """Verify remove_node rejects invalid identifiers."""
        g = NetworkGraph()
        with self.assertRaises(InvalidNodeError):
            g.remove_node(42)

    # ------------------------------------------------------------------
    # Removing edges
    # ------------------------------------------------------------------

    def test_remove_existing_edge(self):
        """Verify removing an edge removes it from both endpoints."""
        g = NetworkGraph()
        g.add_edge("A", "B")
        self.assertTrue(g.remove_edge("A", "B"))
        self.assertFalse(g.has_edge("A", "B"))
        self.assertFalse(g.has_edge("B", "A"))
        # Nodes should still exist
        self.assertTrue(g.has_node("A"))
        self.assertTrue(g.has_node("B"))
        self.assertEqual(g.edge_count(), 0)

    def test_remove_nonexistent_edge_returns_false(self):
        """Verify removing a nonexistent edge is safe and returns False."""
        g = NetworkGraph()
        g.add_node("A")
        g.add_node("B")
        self.assertFalse(g.remove_edge("A", "B"))

    def test_remove_edge_missing_node_returns_false(self):
        """Verify removing edge when a node doesn't exist returns False."""
        g = NetworkGraph()
        g.add_node("A")
        self.assertFalse(g.remove_edge("A", "Z"))

    # ------------------------------------------------------------------
    # Neighbors
    # ------------------------------------------------------------------

    def test_neighbors_returns_insertion_order_copy(self):
        """Verify neighbors are returned in edge insertion order and as a copy."""
        g = NetworkGraph()
        g.add_edge("A", "C")
        g.add_edge("A", "B")
        g.add_edge("A", "D")
        result = g.neighbors("A")
        self.assertEqual(result, ["C", "B", "D"])

    def test_neighbors_copy_isolation(self):
        """Verify mutating returned neighbor list does not affect graph internals."""
        g = NetworkGraph()
        g.add_edge("A", "B")
        result = g.neighbors("A")
        result.append("INJECTED")
        self.assertEqual(g.neighbors("A"), ["B"])

    def test_neighbors_nonexistent_node_raises(self):
        """Verify neighbors() raises NodeNotFoundError for unknown node."""
        g = NetworkGraph()
        with self.assertRaises(NodeNotFoundError):
            g.neighbors("ghost")


class TestBFS(unittest.TestCase):
    """Tests for Breadth-First Search traversal."""

    def _build_diamond_graph(self) -> NetworkGraph:
        """
        Builds a diamond-shaped graph with multiple paths:

            A
           / \\
          B   C
           \\ /
            D

        Edges: A-B, A-C, B-D, C-D
        BFS from A (sorted neighbors): A, B, C, D
          - A's neighbors sorted: [B, C] -> enqueue B, C
          - B's neighbors sorted: [A, D] -> A visited, enqueue D
          - C's neighbors sorted: [A, D] -> both visited
          - D processed, no new neighbors
        """
        g = NetworkGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        return g

    def test_bfs_diamond_graph(self):
        """Verify BFS order on diamond graph with multiple paths."""
        g = self._build_diamond_graph()
        self.assertEqual(g.bfs("A"), ["A", "B", "C", "D"])

    def test_bfs_single_node(self):
        """Verify BFS on a single-node graph."""
        g = NetworkGraph()
        g.add_node("X")
        self.assertEqual(g.bfs("X"), ["X"])

    def test_bfs_linear_chain(self):
        """
        Verify BFS on a linear chain: A - B - C - D - E.
        Expected BFS from A: [A, B, C, D, E]
        """
        g = NetworkGraph()
        for a, b in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]:
            g.add_edge(a, b)
        self.assertEqual(g.bfs("A"), ["A", "B", "C", "D", "E"])

    def test_bfs_with_cycle(self):
        """
        Verify BFS handles cycles without infinite loops.

        Graph (triangle):
            A - B
            |   |
            +- C -+

        Edges: A-B, B-C, A-C
        BFS from A: A, B, C (insertion-order neighbors of A: [B, C])
        """
        g = NetworkGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")
        self.assertEqual(g.bfs("A"), ["A", "B", "C"])

    def test_bfs_disconnected_component(self):
        """Verify BFS only visits the connected component of the start node."""
        g = NetworkGraph()
        g.add_edge("A", "B")
        g.add_node("C")  # Isolated node
        result = g.bfs("A")
        self.assertEqual(result, ["A", "B"])
        self.assertNotIn("C", result)

    def test_bfs_unknown_start_raises(self):
        """Verify BFS raises NodeNotFoundError for unknown start node."""
        g = NetworkGraph()
        g.add_node("A")
        with self.assertRaises(NodeNotFoundError):
            g.bfs("ghost")

    def test_bfs_invalid_start_raises(self):
        """Verify BFS raises InvalidNodeError for invalid start identifier."""
        g = NetworkGraph()
        with self.assertRaises(InvalidNodeError):
            g.bfs("")

    def test_bfs_deterministic_across_calls(self):
        """Verify BFS returns identical results on repeated calls."""
        g = self._build_diamond_graph()
        result1 = g.bfs("A")
        result2 = g.bfs("A")
        self.assertEqual(result1, result2)

    def test_bfs_star_topology(self):
        """
        Verify BFS on a star: center H connected to E, C, A, D, B.

        BFS from H (insertion-order neighbors): [H, E, C, A, D, B]
        """
        g = NetworkGraph()
        for leaf in ["E", "C", "A", "D", "B"]:
            g.add_edge("H", leaf)
        self.assertEqual(g.bfs("H"), ["H", "E", "C", "A", "D", "B"])

    def test_bfs_insertion_order_sensitivity(self):
        """Verify BFS follows neighbor insertion order deterministically."""
        g1 = NetworkGraph()
        g1.add_edge("A", "B")
        g1.add_edge("A", "C")
        self.assertEqual(g1.bfs("A"), ["A", "B", "C"])

        g2 = NetworkGraph()
        g2.add_edge("A", "C")
        g2.add_edge("A", "B")
        self.assertEqual(g2.bfs("A"), ["A", "C", "B"])


class TestDFS(unittest.TestCase):
    """Tests for Depth-First Search traversal."""

    def _build_diamond_graph(self) -> NetworkGraph:
        """
        Same diamond as TestBFS:

            A
           / \\
          B   C
           \\ /
            D

        DFS from A (explicit stack, neighbors pushed reverse insertion order):
          Stack: [A]
          Pop A -> visit A. Neighbors: [B, C]. Push reverse: [C, B]
          Stack: [C, B]
          Pop B -> visit B. Neighbors: [A, D]. A visited. Push reverse: [D]
          Stack: [C, D]
          Pop D -> visit D. Neighbors: [B, C]. B visited. Push reverse: [C]
          Stack: [C, C]
          Pop C -> visit C. Neighbors: [A, D]. Both visited.
          Stack: [C]
          Pop C -> already visited, skip.
          Expected: [A, B, D, C]
        """
        g = NetworkGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        return g

    def test_dfs_diamond_graph(self):
        """Verify DFS order on diamond graph with multiple paths."""
        g = self._build_diamond_graph()
        self.assertEqual(g.dfs("A"), ["A", "B", "D", "C"])

    def test_dfs_single_node(self):
        """Verify DFS on a single-node graph."""
        g = NetworkGraph()
        g.add_node("X")
        self.assertEqual(g.dfs("X"), ["X"])

    def test_dfs_linear_chain(self):
        """
        Verify DFS on a linear chain: A - B - C - D - E.
        Expected DFS from A: [A, B, C, D, E]
        """
        g = NetworkGraph()
        for a, b in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]:
            g.add_edge(a, b)
        self.assertEqual(g.dfs("A"), ["A", "B", "C", "D", "E"])

    def test_dfs_with_cycle(self):
        """
        Verify DFS handles cycles without infinite loops.

        Triangle: A-B, B-C, A-C
        DFS from A:
          Stack: [A]
          Pop A -> visit. Neighbors: [B, C]. Push reverse: [C, B]
          Stack: [C, B]
          Pop B -> visit. Neighbors: [A, C]. A visited. Push reverse: [C]
          Stack: [C, C]
          Pop C -> visit. Neighbors: [A, B]. Both visited.
          Stack: [C]
          Pop C -> already visited.
          Expected: [A, B, C]
        """
        g = NetworkGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")
        self.assertEqual(g.dfs("A"), ["A", "B", "C"])

    def test_dfs_disconnected_component(self):
        """Verify DFS only visits the connected component of the start node."""
        g = NetworkGraph()
        g.add_edge("A", "B")
        g.add_node("C")  # Isolated
        result = g.dfs("A")
        self.assertEqual(result, ["A", "B"])
        self.assertNotIn("C", result)

    def test_dfs_unknown_start_raises(self):
        """Verify DFS raises NodeNotFoundError for unknown start node."""
        g = NetworkGraph()
        g.add_node("A")
        with self.assertRaises(NodeNotFoundError):
            g.dfs("ghost")

    def test_dfs_invalid_start_raises(self):
        """Verify DFS raises InvalidNodeError for invalid start identifier."""
        g = NetworkGraph()
        with self.assertRaises(InvalidNodeError):
            g.dfs(None)

    def test_dfs_deterministic_across_calls(self):
        """Verify DFS returns identical results on repeated calls."""
        g = self._build_diamond_graph()
        result1 = g.dfs("A")
        result2 = g.dfs("A")
        self.assertEqual(result1, result2)

    def test_dfs_deep_graph_no_recursion_limit(self):
        """
        Verify DFS with an explicit stack handles a graph deeper than
        Python's default recursion limit (typically 1000).

        Builds a linear chain of 1500 nodes: node-0 - node-1 - ... - node-1499.
        A naive recursive DFS would hit RecursionError; our stack-based DFS must not.
        """
        g = NetworkGraph()
        depth = 1500
        for i in range(depth - 1):
            g.add_edge(f"node-{i}", f"node-{i + 1}")

        # Should NOT raise RecursionError
        result = g.dfs("node-0")

        self.assertEqual(len(result), depth)
        self.assertEqual(result[0], "node-0")
        self.assertEqual(result[-1], f"node-{depth - 1}")

    def test_dfs_star_topology(self):
        """
        Verify DFS on a star: center H connected to E, C, A, D, B in that order.

        DFS from H:
          Stack: [H]
          Pop H -> visit. Neighbors in insertion order: [E, C, A, D, B].
          Push in reverse order: [B, D, A, C, E] (so E is popped next).
          Pop E -> visit. Neighbors: [H] (visited). Nothing pushed.
          Pop C -> visit. Same.
          Pop A -> visit. Same.
          Pop D -> visit. Same.
          Pop B -> visit. Same.
          Expected visitation order: [H, E, C, A, D, B]
        """
        g = NetworkGraph()
        for leaf in ["E", "C", "A", "D", "B"]:
            g.add_edge("H", leaf)
        self.assertEqual(g.dfs("H"), ["H", "E", "C", "A", "D", "B"])

    def test_dfs_insertion_order_sensitivity(self):
        """Verify DFS visits first-inserted neighbor first."""
        g1 = NetworkGraph()
        g1.add_edge("A", "B")
        g1.add_edge("A", "C")
        self.assertEqual(g1.dfs("A"), ["A", "B", "C"])

        g2 = NetworkGraph()
        g2.add_edge("A", "C")
        g2.add_edge("A", "B")
        self.assertEqual(g2.dfs("A"), ["A", "C", "B"])


class TestBFSTree(unittest.TestCase):
    """Tests for BFS parent mapping (bfs_tree)."""

    def test_bfs_tree_diamond(self):
        """
        BFS tree of the diamond graph from A.

            A
           / \\
          B   C
           \\ /
            D

        BFS from A: A (parent=None), B (parent=A), C (parent=A), D (parent=B)
        D discovered via B before C because B is dequeued first.
        """
        g = NetworkGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")

        tree = g.bfs_tree("A")
        self.assertIsNone(tree["A"])
        self.assertEqual(tree["B"], "A")
        self.assertEqual(tree["C"], "A")
        self.assertEqual(tree["D"], "B")

    def test_bfs_tree_single_node(self):
        """Verify BFS tree of a single node maps it to None."""
        g = NetworkGraph()
        g.add_node("X")
        tree = g.bfs_tree("X")
        self.assertEqual(tree, {"X": None})

    def test_bfs_tree_disconnected(self):
        """Verify BFS tree only contains nodes in the start's component."""
        g = NetworkGraph()
        g.add_edge("A", "B")
        g.add_node("C")
        tree = g.bfs_tree("A")
        self.assertIn("A", tree)
        self.assertIn("B", tree)
        self.assertNotIn("C", tree)

    def test_bfs_tree_unknown_start_raises(self):
        """Verify bfs_tree raises NodeNotFoundError for unknown start."""
        g = NetworkGraph()
        with self.assertRaises(NodeNotFoundError):
            g.bfs_tree("ghost")


class TestDFSTree(unittest.TestCase):
    """Tests for DFS parent mapping (dfs_tree)."""

    def test_dfs_tree_diamond(self):
        """
        DFS tree of the diamond graph from A.

        DFS visit order: A, B, D, C
          A (parent=None)
          B (parent=A)
          D (parent=B)
          C (parent=D or A? Let's trace:)

        Stack: [(A, None)]
        Pop (A, None) -> parent[A] = None. Push neighbors in reverse insertion order:
          neighbors [B, C] -> reverse [C, B] -> push (C, A), (B, A)
        Stack: [(C, A), (B, A)]
        Pop (B, A) -> parent[B] = A. Push neighbors in reverse insertion order:
          neighbors [A, D] -> reverse [D, A]. A in parent, push (D, B)
        Stack: [(C, A), (D, B)]
        Pop (D, B) -> parent[D] = B. Push neighbors in reverse insertion order:
          neighbors [B, C] -> reverse [C, B]. B in parent, push (C, D)
        Stack: [(C, A), (C, D)]
        Pop (C, D) -> parent[C] = D. Push neighbors in reverse insertion order:
          neighbors [A, D] -> reverse [D, A]. Both in parent.
        Stack: [(C, A)]
        Pop (C, A) -> C already in parent. Skip.

        Result: A->None, B->A, D->B, C->D
        """
        g = NetworkGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")

        tree = g.dfs_tree("A")
        self.assertIsNone(tree["A"])
        self.assertEqual(tree["B"], "A")
        self.assertEqual(tree["D"], "B")
        self.assertEqual(tree["C"], "D")

    def test_dfs_tree_single_node(self):
        """Verify DFS tree of a single node maps it to None."""
        g = NetworkGraph()
        g.add_node("X")
        tree = g.dfs_tree("X")
        self.assertEqual(tree, {"X": None})

    def test_dfs_tree_disconnected(self):
        """Verify DFS tree only contains nodes in the start's component."""
        g = NetworkGraph()
        g.add_edge("A", "B")
        g.add_node("C")
        tree = g.dfs_tree("A")
        self.assertIn("A", tree)
        self.assertIn("B", tree)
        self.assertNotIn("C", tree)

    def test_dfs_tree_unknown_start_raises(self):
        """Verify dfs_tree raises NodeNotFoundError for unknown start."""
        g = NetworkGraph()
        with self.assertRaises(NodeNotFoundError):
            g.dfs_tree("ghost")


if __name__ == "__main__":
    unittest.main()
