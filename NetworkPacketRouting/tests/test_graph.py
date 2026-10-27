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

import math
import sys
import unittest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))

from NetworkPacketRouting.graph import (
    DistanceOverflowError,
    EdgeNotFoundError,
    GraphError,
    InvalidNodeError,
    InvalidWeightError,
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


class TestNetworkGraphWeights(unittest.TestCase):
    """Tests for weighted edges in NetworkGraph."""

    def test_default_weight_is_one(self):
        """Verify unweighted add_edge defaults weight to 1."""
        g = NetworkGraph()
        g.add_edge("A", "B")
        self.assertEqual(g.get_edge_weight("A", "B"), 1)
        self.assertEqual(g.get_edge_weight("B", "A"), 1)
        self.assertEqual(g.edge_weight("A", "B"), 1)

    def test_explicit_integer_and_float_weights(self):
        """Verify integer and floating-point weights are preserved symmetrically."""
        g = NetworkGraph()
        g.add_edge("A", "B", weight=5)
        g.add_edge("B", "C", weight=2.75)
        self.assertEqual(g.get_edge_weight("A", "B"), 5)
        self.assertEqual(g.get_edge_weight("B", "A"), 5)
        self.assertEqual(g.get_edge_weight("B", "C"), 2.75)
        self.assertEqual(g.get_edge_weight("C", "B"), 2.75)

    def test_zero_weight_edge(self):
        """Verify zero-weight edges (both int 0 and float 0.0) are valid."""
        g = NetworkGraph()
        g.add_edge("A", "B", weight=0)
        g.add_edge("B", "C", weight=0.0)
        self.assertEqual(g.get_edge_weight("A", "B"), 0)
        self.assertEqual(g.get_edge_weight("B", "C"), 0.0)

    def test_arbitrary_precision_large_integer_weight_accepted(self):
        """Verify arbitrarily large non-negative Python integers are valid weights."""
        g = NetworkGraph()
        large_int = 10**1000
        g.add_edge("A", "B", weight=large_int)
        self.assertEqual(g.get_edge_weight("A", "B"), large_int)
        self.assertEqual(g.get_edge_weight("B", "A"), large_int)

        # Also test updating with large int
        g.set_edge_weight("A", "B", 10**2000)
        self.assertEqual(g.get_edge_weight("A", "B"), 10**2000)

    def test_arbitrary_precision_large_negative_integer_rejected(self):
        """Verify arbitrarily large negative Python integers raise InvalidWeightError."""
        g = NetworkGraph()
        large_neg = -(10**1000)
        with self.assertRaises(InvalidWeightError):
            g.add_edge("A", "B", weight=large_neg)

        g.add_edge("A", "B", weight=1)
        with self.assertRaises(InvalidWeightError):
            g.set_edge_weight("A", "B", large_neg)

    def test_negative_weight_rejected(self):
        """Verify negative weights raise InvalidWeightError."""
        g = NetworkGraph()
        for neg in [-1, -0.001, -100.5, -(10**50)]:
            with self.assertRaises(InvalidWeightError, msg=f"Should reject {neg}"):
                g.add_edge("A", "B", weight=neg)

    def test_boolean_weight_rejected(self):
        """Verify booleans are rejected as weights even though bool inherits from int."""
        g = NetworkGraph()
        with self.assertRaises(InvalidWeightError):
            g.add_edge("A", "B", weight=True)
        with self.assertRaises(InvalidWeightError):
            g.add_edge("A", "B", weight=False)

    def test_nan_and_inf_weight_rejected(self):
        """Verify NaN, inf, and -inf are rejected as weights."""
        g = NetworkGraph()
        with self.assertRaises(InvalidWeightError):
            g.add_edge("A", "B", weight=float("nan"))
        with self.assertRaises(InvalidWeightError):
            g.add_edge("A", "B", weight=float("inf"))
        with self.assertRaises(InvalidWeightError):
            g.add_edge("A", "B", weight=float("-inf"))

    def test_non_numeric_weight_rejected(self):
        """Verify non-numeric types are rejected as weights."""
        g = NetworkGraph()
        for bad in ["1", "5.0", None, [1], {"w": 1}]:
            with self.assertRaises(InvalidWeightError, msg=f"Should reject {bad!r}"):
                g.add_edge("A", "B", weight=bad)  # type: ignore

    def test_get_edge_weight_unknown_node_raises(self):
        """Verify get_edge_weight raises NodeNotFoundError if either node is unknown."""
        g = NetworkGraph()
        g.add_edge("A", "B", weight=10)
        with self.assertRaises(NodeNotFoundError):
            g.get_edge_weight("A", "ghost")
        with self.assertRaises(NodeNotFoundError):
            g.get_edge_weight("ghost", "B")
        with self.assertRaises(NodeNotFoundError):
            g.get_edge_weight("ghost1", "ghost2")

    def test_get_edge_weight_nonexistent_edge_raises(self):
        """Verify get_edge_weight raises EdgeNotFoundError if both nodes exist but no edge exists."""
        g = NetworkGraph()
        g.add_node("A")
        g.add_node("B")
        with self.assertRaises(EdgeNotFoundError):
            g.get_edge_weight("A", "B")
        with self.assertRaises(EdgeNotFoundError):
            g.get_edge_weight("B", "A")

    def test_get_edge_weight_invalid_node_id_raises(self):
        """Verify get_edge_weight validates node identifiers and raises InvalidNodeError."""
        g = NetworkGraph()
        g.add_edge("A", "B", 1)
        with self.assertRaises(InvalidNodeError):
            g.get_edge_weight("", "B")
        with self.assertRaises(InvalidNodeError):
            g.get_edge_weight("A", " ")
        with self.assertRaises(InvalidNodeError):
            g.get_edge_weight(123, "B")  # type: ignore

    def test_set_edge_weight_updates_symmetrically(self):
        """Verify set_edge_weight and update_edge_weight update weight in both directions."""
        g = NetworkGraph()
        g.add_edge("A", "B", weight=1)
        self.assertEqual(g.get_edge_weight("A", "B"), 1)

        g.set_edge_weight("A", "B", 7.5)
        self.assertEqual(g.get_edge_weight("A", "B"), 7.5)
        self.assertEqual(g.get_edge_weight("B", "A"), 7.5)

        g.update_edge_weight("B", "A", 42)
        self.assertEqual(g.get_edge_weight("A", "B"), 42)
        self.assertEqual(g.get_edge_weight("B", "A"), 42)

    def test_set_edge_weight_validation(self):
        """Verify set_edge_weight rejects invalid weights, unknown nodes, and nonexistent edges."""
        g = NetworkGraph()
        g.add_edge("A", "B", weight=1)
        g.add_node("C")

        with self.assertRaises(InvalidWeightError):
            g.set_edge_weight("A", "B", -1)
        with self.assertRaises(InvalidWeightError):
            g.set_edge_weight("A", "B", True)
        with self.assertRaises(InvalidWeightError):
            g.set_edge_weight("A", "B", float("nan"))
        with self.assertRaises(NodeNotFoundError):
            g.set_edge_weight("A", "ghost", 5)
        with self.assertRaises(EdgeNotFoundError):
            g.set_edge_weight("A", "C", 5)

    def test_set_edge_weight_preserves_neighbor_order(self):
        """Verify updating an edge weight preserves the deterministic insertion order of neighbors."""
        g = NetworkGraph()
        g.add_edge("A", "B", weight=1)
        g.add_edge("A", "C", weight=2)
        g.add_edge("A", "D", weight=3)
        self.assertEqual(g.neighbors("A"), ["B", "C", "D"])

        # Update weight of the first neighbor
        g.set_edge_weight("A", "B", 999)
        self.assertEqual(g.neighbors("A"), ["B", "C", "D"])
        self.assertEqual(g.get_edge_weight("A", "B"), 999)

        # Update weight of the middle neighbor
        g.set_edge_weight("A", "C", 888)
        self.assertEqual(g.neighbors("A"), ["B", "C", "D"])
        self.assertEqual(g.get_edge_weight("A", "C"), 888)

    def test_duplicate_add_edge_does_not_overwrite_weight(self):
        """Verify calling add_edge on an existing edge returns False and preserves original weight."""
        g = NetworkGraph()
        self.assertTrue(g.add_edge("A", "B", weight=10))
        self.assertFalse(g.add_edge("A", "B", weight=20))
        self.assertFalse(g.add_edge("B", "A", weight=30))
        self.assertEqual(g.get_edge_weight("A", "B"), 10)
        self.assertEqual(g.get_edge_weight("B", "A"), 10)

    def test_bfs_dfs_unaffected_by_weights(self):
        """
        Verify BFS and DFS traversal orders remain strictly deterministic and
        governed by edge insertion order, completely unaffected by edge weights.
        """
        # Topology: A connected to B (wt=100) then C (wt=1).
        # B connected to D (wt=50), C connected to D (wt=2).
        g = NetworkGraph()
        g.add_edge("A", "B", weight=100)
        g.add_edge("A", "C", weight=1)
        g.add_edge("B", "D", weight=50)
        g.add_edge("C", "D", weight=2)

        # BFS explores in insertion order: A -> B, C -> D (via B)
        self.assertEqual(g.bfs("A"), ["A", "B", "C", "D"])
        bfs_tree = g.bfs_tree("A")
        self.assertEqual(bfs_tree["D"], "B")  # Discovered through B first

        # DFS explores first-inserted first: A -> B -> D -> C
        self.assertEqual(g.dfs("A"), ["A", "B", "D", "C"])

    def test_weighted_graph_cycles_and_disconnected(self):
        """Verify cycles and disconnected components work identically with weights."""
        g = NetworkGraph()
        # Component 1: triangle cycle
        g.add_edge("A", "B", weight=3.5)
        g.add_edge("B", "C", weight=4.0)
        g.add_edge("C", "A", weight=1.2)
        # Component 2: isolated weighted edge
        g.add_edge("X", "Y", weight=0.5)

        self.assertEqual(g.node_count(), 5)
        self.assertEqual(g.edge_count(), 4)
        self.assertEqual(g.get_edge_weight("A", "C"), 1.2)
        self.assertEqual(g.get_edge_weight("X", "Y"), 0.5)
        self.assertEqual(g.bfs("A"), ["A", "B", "C"])
        self.assertEqual(g.bfs("X"), ["X", "Y"])

    def test_weighted_deep_linear_graph(self):
        """Verify deep graph with weights handles 1000+ nodes without recursion limits."""
        g = NetworkGraph()
        chain_len = 1000
        for i in range(chain_len - 1):
            g.add_edge(f"N{i}", f"N{i+1}", weight=float(i % 10))

        self.assertEqual(g.node_count(), chain_len)
        self.assertEqual(g.edge_count(), chain_len - 1)
        self.assertEqual(g.get_edge_weight("N0", "N1"), 0.0)
        self.assertEqual(g.get_edge_weight("N9", "N10"), 9.0)

        dfs_order = g.dfs("N0")
        self.assertEqual(len(dfs_order), chain_len)
        self.assertEqual(dfs_order[0], "N0")
        self.assertEqual(dfs_order[-1], f"N{chain_len - 1}")


class TestNetworkGraphDijkstra(unittest.TestCase):
    """Tests for Dijkstra's shortest-path algorithm and weighted pathfinding."""

    def test_minimum_cost_path_has_more_hops_than_minimum_hop_path(self):
        """
        Verify that Dijkstra selects a lower-cost multi-hop path over a higher-cost direct link.

        Topology:
            A --(10)--> B (direct link, 1 hop, cost 10)
            A --(1)--> C --(2)--> D --(1)--> B (multi-hop path, 3 hops, cost 4)
        """
        g = NetworkGraph()
        g.add_edge("A", "B", weight=10)
        g.add_edge("A", "C", weight=1)
        g.add_edge("C", "D", weight=2)
        g.add_edge("D", "B", weight=1)

        # Unweighted BFS chooses direct link (1 hop)
        self.assertEqual(g.bfs("A")[:2], ["A", "B"])

        # Dijkstra chooses multi-hop path (3 hops, cost 4)
        result = g.shortest_path_weighted("A", "B")
        self.assertIsNotNone(result)
        path, cost = result
        self.assertEqual(path, ["A", "C", "D", "B"])
        self.assertEqual(cost, 4)

    def test_dijkstra_known_graph_distances_and_predecessors(self):
        """
        Verify Dijkstra distance and predecessor mappings against a known 5-node graph.

        Edges:
            A-B: 4, A-C: 2, C-B: 1, B-D: 5, C-D: 8, D-E: 2
        Expected from A:
            dist[A]=0, pred: none
            dist[C]=2, pred[C]=A
            dist[B]=3 (A-C-B), pred[B]=C
            dist[D]=8 (A-C-B-D), pred[D]=B
            dist[E]=10 (A-C-B-D-E), pred[E]=D
        """
        g = NetworkGraph()
        g.add_edge("A", "B", weight=4)
        g.add_edge("A", "C", weight=2)
        g.add_edge("C", "B", weight=1)
        g.add_edge("B", "D", weight=5)
        g.add_edge("C", "D", weight=8)
        g.add_edge("D", "E", weight=2)

        distances, predecessors = g.dijkstra("A")

        self.assertEqual(distances["A"], 0)
        self.assertEqual(distances["C"], 2)
        self.assertEqual(distances["B"], 3)
        self.assertEqual(distances["D"], 8)
        self.assertEqual(distances["E"], 10)

        # Source has no predecessor
        self.assertNotIn("A", predecessors)
        self.assertEqual(predecessors["C"], "A")
        self.assertEqual(predecessors["B"], "C")
        self.assertEqual(predecessors["D"], "B")
        self.assertEqual(predecessors["E"], "D")

    def test_shortest_path_weighted_reconstruction(self):
        """Verify path reconstruction produces the complete node sequence and exact cost."""
        g = NetworkGraph()
        g.add_edge("A", "B", weight=4)
        g.add_edge("A", "C", weight=2)
        g.add_edge("C", "B", weight=1)
        g.add_edge("B", "D", weight=5)
        g.add_edge("D", "E", weight=2)

        self.assertEqual(
            g.shortest_path_weighted("A", "E"),
            (["A", "C", "B", "D", "E"], 10),
        )
        self.assertEqual(
            g.shortest_path_weighted("A", "B"),
            (["A", "C", "B"], 3),
        )
        self.assertEqual(
            g.shortest_path_weighted("C", "E"),
            (["C", "B", "D", "E"], 8),
        )

    def test_same_source_and_target(self):
        """Verify querying path from a node to itself returns ([source], 0)."""
        g = NetworkGraph()
        g.add_edge("A", "B", weight=5)
        g.add_node("X")

        self.assertEqual(g.shortest_path_weighted("A", "A"), (["A"], 0))
        self.assertEqual(g.shortest_path_weighted("X", "X"), (["X"], 0))

    def test_unreachable_target_and_disconnected_components(self):
        """
        Verify that unreachable targets map to math.inf in dijkstra() with no
        predecessor, and shortest_path_weighted() returns None.
        """
        g = NetworkGraph()
        # Component 1: A-B
        g.add_edge("A", "B", weight=2)
        # Component 2: C-D
        g.add_edge("C", "D", weight=3)
        # Isolated node: Z
        g.add_node("Z")

        distances, predecessors = g.dijkstra("A")

        # Every node in the graph is in distances
        self.assertEqual(len(distances), 5)
        self.assertEqual(distances["A"], 0)
        self.assertEqual(distances["B"], 2)
        self.assertEqual(distances["C"], math.inf)
        self.assertEqual(distances["D"], math.inf)
        self.assertEqual(distances["Z"], math.inf)

        # Source and unreachable nodes have no predecessor
        self.assertNotIn("A", predecessors)
        self.assertEqual(predecessors["B"], "A")
        self.assertNotIn("C", predecessors)
        self.assertNotIn("D", predecessors)
        self.assertNotIn("Z", predecessors)

        # shortest_path_weighted returns None for unreachable targets
        self.assertIsNone(g.shortest_path_weighted("A", "C"))
        self.assertIsNone(g.shortest_path_weighted("A", "D"))
        self.assertIsNone(g.shortest_path_weighted("A", "Z"))
        self.assertIsNone(g.shortest_path_weighted("C", "A"))

    def test_zero_weight_edges_and_zero_weight_cycles(self):
        """
        Verify that zero-weight edges and zero-weight cycles are handled without
        infinite loops, distance degradation, or predecessor cycles.
        """
        g = NetworkGraph()
        # Zero-weight triangle cycle
        g.add_edge("A", "B", weight=0)
        g.add_edge("B", "C", weight=0)
        g.add_edge("C", "A", weight=0)
        # Extension with weight
        g.add_edge("C", "D", weight=5)
        g.add_edge("D", "E", weight=0)

        distances, predecessors = g.dijkstra("A")

        self.assertEqual(distances["A"], 0)
        self.assertEqual(distances["B"], 0)
        self.assertEqual(distances["C"], 0)
        self.assertEqual(distances["D"], 5)
        self.assertEqual(distances["E"], 5)

        # Source A has no predecessor
        self.assertNotIn("A", predecessors)

        # Reconstructed path must be valid and acyclic
        result = g.shortest_path_weighted("A", "E")
        self.assertIsNotNone(result)
        path, cost = result
        self.assertEqual(cost, 5)
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "E")
        # Ensure path contains no duplicate nodes (no cycle)
        self.assertEqual(len(path), len(set(path)))

    def test_floating_point_edge_weights(self):
        """Verify Dijkstra supports float weights with exact fractional sums."""
        g = NetworkGraph()
        g.add_edge("A", "B", weight=1.25)
        g.add_edge("B", "C", weight=2.5)
        g.add_edge("A", "C", weight=4.0)

        result = g.shortest_path_weighted("A", "C")
        self.assertIsNotNone(result)
        path, cost = result
        self.assertEqual(path, ["A", "B", "C"])
        self.assertEqual(cost, 3.75)

    def test_equal_cost_alternatives_deterministic_results(self):
        """
        Verify equal-cost alternative routes produce deterministic results
        across repeated invocations.
        """
        g = NetworkGraph()
        # Diamond topology with identical weights on both sides
        g.add_edge("A", "B", weight=3)
        g.add_edge("A", "C", weight=3)
        g.add_edge("B", "D", weight=2)
        g.add_edge("C", "D", weight=2)

        # Because B was inserted before C, A->B is explored first and settled first
        first_run = g.shortest_path_weighted("A", "D")
        self.assertIsNotNone(first_run)

        for _ in range(10):
            subsequent_run = g.shortest_path_weighted("A", "D")
            self.assertEqual(subsequent_run, first_run)

    def test_invalid_and_nonexistent_nodes_raise(self):
        """Verify dijkstra and shortest_path_weighted validate node arguments."""
        g = NetworkGraph()
        g.add_edge("A", "B", weight=1)

        # Invalid node identifiers
        for bad in ["", " ", None, 123, " A"]:
            with self.assertRaises(InvalidNodeError):
                g.dijkstra(bad)  # type: ignore
            with self.assertRaises(InvalidNodeError):
                g.shortest_path_weighted(bad, "B")  # type: ignore
            with self.assertRaises(InvalidNodeError):
                g.shortest_path_weighted("A", bad)  # type: ignore

        # Nonexistent nodes
        with self.assertRaises(NodeNotFoundError):
            g.dijkstra("ghost")
        with self.assertRaises(NodeNotFoundError):
            g.shortest_path_weighted("ghost", "B")
        with self.assertRaises(NodeNotFoundError):
            g.shortest_path_weighted("A", "ghost")

    def test_arbitrary_precision_large_integer_path_costs(self):
        """
        Verify exact integer arithmetic is preserved for integer-only paths
        with very large numbers without converting to float or losing precision.
        """
        g = NetworkGraph()
        w1 = 10**500
        w2 = 2 * (10**500)
        g.add_edge("A", "B", weight=w1)
        g.add_edge("B", "C", weight=w2)

        distances, _ = g.dijkstra("A")
        self.assertEqual(distances["C"], 3 * (10**500))
        self.assertIsInstance(distances["C"], int)

        path_result = g.shortest_path_weighted("A", "C")
        self.assertIsNotNone(path_result)
        path, cost = path_result
        self.assertEqual(path, ["A", "B", "C"])
        self.assertEqual(cost, 3 * (10**500))
        self.assertIsInstance(cost, int)

    def test_floating_point_overflow_raises_distance_overflow_error(self):
        """
        Verify that when finite float edge weights sum to infinity through float overflow,
        DistanceOverflowError is explicitly raised rather than treating the node as disconnected.
        """
        g = NetworkGraph()
        g.add_edge("A", "B", weight=1e308)
        g.add_edge("B", "C", weight=1e308)

        with self.assertRaises(DistanceOverflowError):
            g.dijkstra("A")

        with self.assertRaises(DistanceOverflowError):
            g.shortest_path_weighted("A", "C")

    def test_dijkstra_does_not_mutate_graph(self):
        """Verify calling dijkstra and shortest_path_weighted leaves the graph unchanged."""
        g = NetworkGraph()
        g.add_edge("A", "B", weight=3)
        g.add_edge("B", "C", weight=4)
        g.add_edge("A", "C", weight=10)

        nodes_before = g.nodes()
        edges_before = g.edge_count()
        neighbors_a_before = g.neighbors("A")
        weight_ab_before = g.get_edge_weight("A", "B")

        g.dijkstra("A")
        g.shortest_path_weighted("A", "C")

        self.assertEqual(g.nodes(), nodes_before)
        self.assertEqual(g.edge_count(), edges_before)
        self.assertEqual(g.neighbors("A"), neighbors_a_before)
        self.assertEqual(g.get_edge_weight("A", "B"), weight_ab_before)


if __name__ == "__main__":
    unittest.main()
