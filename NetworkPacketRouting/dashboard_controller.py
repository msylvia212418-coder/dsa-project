"""
dashboard_controller.py - Controller and simulation support for the Network Packet Routing Dashboard.

This module provides pure-logic services for the interactive dashboard:
1. Demo topology builder with classical Dijkstra vs BFS trade-offs.
2. Route comparison service evaluating BFS (shortest-hop) vs Dijkstra (least-cost).
3. In-process packet forwarding simulation with step-by-step TTL and status tracking.
4. Graphviz DOT string generator for visual topology rendering.
5. Topology metrics aggregator.

Decoupled from Streamlit or UI rendering, fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from NetworkPacketRouting.graph import DEFAULT_WEIGHT, InvalidWeightError
from NetworkPacketRouting.packet import Packet
from NetworkPacketRouting.route_planner import (
    build_routing_table,
    build_weighted_routing_table,
)
from NetworkPacketRouting.routing_table import RoutingTable
from NetworkPacketRouting.topology import (
    LinkNotFoundError,
    NetworkTopology,
    NodeNotFoundError,
)


@dataclass(frozen=True)
class RouteComparisonResult:
    """Result of comparing BFS and Dijkstra paths between two nodes."""

    source: str
    destination: str
    bfs_path: Optional[List[str]]
    bfs_hops: Optional[int]
    bfs_cost: Optional[Union[int, float]]
    bfs_next_hop: Optional[str]
    dijkstra_path: Optional[List[str]]
    dijkstra_hops: Optional[int]
    dijkstra_cost: Optional[Union[int, float]]
    dijkstra_next_hop: Optional[str]
    is_different_path: bool
    is_cost_divergence: bool


@dataclass
class SimulationStep:
    """A single hop in a packet's forwarding journey."""

    hop_number: int
    current_node: str
    incoming_ttl: int
    outgoing_ttl: int
    action: str
    next_hop: Optional[str]
    status: str
    detail: str


@dataclass
class SimulationResult:
    """Comprehensive result of an in-process packet forwarding simulation."""

    packet_id: str
    source: str
    destination: str
    payload: str
    initial_ttl: int
    final_ttl: int
    routing_mode: str
    status: str  # "DELIVERED", "DROPPED_TTL", "NO_ROUTE", "UNREACHABLE"
    hops_traversed: List[str]
    steps: List[SimulationStep]
    delivered: bool
    summary: str


def create_demo_topology() -> NetworkTopology:
    """
    Creates the standard demo network topology with a classic diamond structure.

    Topology Layout:
        R1 --[100]-- R2 (expensive direct link: 1 hop, cost 100)
        R1 --[2]---- R3 --[2]---- R2 (cheap multi-hop: 2 hops, cost 4)
        R1 --[5]---- R4 --[5]---- R2 (alternate multi-hop: 2 hops, cost 10)
        R2 --[1]---- Dest-Alpha (receiver endpoint)
        R4 --[2]---- Dest-Beta (receiver endpoint)

    Educational properties:
        - R1 -> Dest-Alpha:
            BFS selects:     R1 -> R2 -> Dest-Alpha (2 hops, cost 101)
            Dijkstra selects: R1 -> R3 -> R2 -> Dest-Alpha (3 hops, cost 5)
            Dramatically illustrates why least-hop is NOT least-cost.
        - R1 -> Dest-Beta:
            BFS selects:     R1 -> R4 -> Dest-Beta (2 hops, cost 7)
            Dijkstra selects: R1 -> R4 -> Dest-Beta (2 hops, cost 7)
            Demonstrates cases where BFS and Dijkstra converge.
    """
    topo = NetworkTopology()

    # Register Routers
    topo.register_node("R1", "127.0.0.1", 5001)
    topo.register_node("R2", "127.0.0.1", 5002)
    topo.register_node("R3", "127.0.0.1", 5003)
    topo.register_node("R4", "127.0.0.1", 5004)

    # Register Receiver Endpoints (leaf destination nodes)
    topo.register_node("Dest-Alpha", "127.0.0.1", 6001)
    topo.register_node("Dest-Beta", "127.0.0.1", 6002)

    # Core Router Mesh
    topo.connect("R1", "R2", weight=100)  # Expensive direct link
    topo.connect("R1", "R3", weight=2)    # Cheap path via R3
    topo.connect("R3", "R2", weight=2)    # Cost R1->R3->R2 = 4
    topo.connect("R1", "R4", weight=5)    # Alternative path via R4
    topo.connect("R4", "R2", weight=5)    # Cost R1->R4->R2 = 10

    # Leaf Links to Receivers
    topo.connect("R2", "Dest-Alpha", weight=1)
    topo.connect("R4", "Dest-Beta", weight=2)

    return topo


def get_topology_metrics(topology: NetworkTopology) -> Dict[str, Any]:
    """
    Aggregates live structural metrics of the topology.

    Returns:
        Dictionary with node count, link count, router count, receiver count,
        total weight, and lists of node categories.
    """
    nodes = topology.nodes()
    routers = [n for n in nodes if not n.startswith("Dest-") and not n.startswith("receiver")]
    receivers = [n for n in nodes if n.startswith("Dest-") or n.startswith("receiver")]

    links: List[Tuple[str, str, Union[int, float]]] = []
    seen_links: Set[Tuple[str, str]] = set()
    total_weight: Union[int, float] = 0

    for u in nodes:
        for v in topology.neighbors(u):
            pair = (min(u, v), max(u, v))
            if pair not in seen_links:
                seen_links.add(pair)
                w = topology.get_link_weight(u, v)
                links.append((pair[0], pair[1], w))
                total_weight += w

    return {
        "node_count": len(nodes),
        "router_count": len(routers),
        "receiver_count": len(receivers),
        "link_count": len(links),
        "total_weight": total_weight,
        "routers": routers,
        "receivers": receivers,
        "nodes": nodes,
        "links": links,
    }


def compute_path_cost(topology: NetworkTopology, path: List[str]) -> Union[int, float]:
    """
    Computes the total sum of weights along a given path in the topology.
    Returns 0 if path has <= 1 node.
    """
    if len(path) <= 1:
        return 0

    total: Union[int, float] = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        total += topology.get_link_weight(u, v)
    return total


def compare_routes(
    topology: NetworkTopology, source: str, destination: str
) -> RouteComparisonResult:
    """
    Compares the routes chosen by Unweighted BFS and Weighted Dijkstra.

    Args:
        topology: The active NetworkTopology instance.
        source: Starting node identifier.
        destination: Target node identifier.

    Returns:
        RouteComparisonResult containing side-by-side metrics.
    """
    if not topology.has_node(source):
        raise NodeNotFoundError(f"Source node {source!r} is not registered in the topology")
    if not topology.has_node(destination):
        raise NodeNotFoundError(f"Destination node {destination!r} is not registered in the topology")

    # 1. Unweighted BFS
    bfs_path = topology.shortest_path(source, destination)
    bfs_hops: Optional[int] = None
    bfs_cost: Optional[Union[int, float]] = None
    bfs_next_hop: Optional[str] = None

    if bfs_path is not None:
        bfs_hops = len(bfs_path) - 1
        bfs_cost = compute_path_cost(topology, bfs_path)
        bfs_next_hop = bfs_path[1] if len(bfs_path) > 1 else None

    # 2. Weighted Dijkstra
    dijkstra_res = topology.shortest_path_weighted(source, destination)
    dijkstra_path: Optional[List[str]] = None
    dijkstra_hops: Optional[int] = None
    dijkstra_cost: Optional[Union[int, float]] = None
    dijkstra_next_hop: Optional[str] = None

    if dijkstra_res is not None:
        dijkstra_path, dijkstra_cost = dijkstra_res
        dijkstra_hops = len(dijkstra_path) - 1
        dijkstra_next_hop = dijkstra_path[1] if len(dijkstra_path) > 1 else None

    is_diff = (bfs_path != dijkstra_path)
    is_cost_div = (
        bfs_cost is not None
        and dijkstra_cost is not None
        and bfs_cost != dijkstra_cost
    )

    return RouteComparisonResult(
        source=source,
        destination=destination,
        bfs_path=bfs_path,
        bfs_hops=bfs_hops,
        bfs_cost=bfs_cost,
        bfs_next_hop=bfs_next_hop,
        dijkstra_path=dijkstra_path,
        dijkstra_hops=dijkstra_hops,
        dijkstra_cost=dijkstra_cost,
        dijkstra_next_hop=dijkstra_next_hop,
        is_different_path=is_diff,
        is_cost_divergence=is_cost_div,
    )


def simulate_packet_forwarding(
    topology: NetworkTopology,
    source: str,
    destination: str,
    payload: str = "Demo Payload",
    initial_ttl: int = 8,
    routing_mode: str = "weighted",
    max_hops: int = 50,
) -> SimulationResult:
    """
    Simulates hop-by-hop packet forwarding across the topology using the project's
    exact packet and routing table semantics.

    At each router along the path:
    1. Inspects destination.
    2. Checks if destination is the current node (DELIVERED).
    3. Checks TTL: if TTL <= 1, packet is dropped (DROPPED_TTL).
    4. Performs route lookup in the router's compiled routing table.
       If no route exists, packet is dropped (DROPPED_NO_ROUTE).
    5. Decrements TTL and forwards to next hop.

    Args:
        topology: The active NetworkTopology instance.
        source: Origin node identifier.
        destination: Final target node identifier.
        payload: String data payload.
        initial_ttl: Initial Time-To-Live integer.
        routing_mode: 'unweighted' (BFS) or 'weighted' (Dijkstra).
        max_hops: Loop guard limit.

    Returns:
        SimulationResult detailing every hop step, TTL values, and final outcome.
    """
    if not topology.has_node(source):
        raise NodeNotFoundError(f"Source {source!r} is not in topology")
    if not topology.has_node(destination):
        raise NodeNotFoundError(f"Destination {destination!r} is not in topology")

    packet = Packet(
        version=1,
        src=source,
        dest=destination,
        payload=payload,
        packet_id="sim-pkt-1",
        ttl=initial_ttl,
    )

    steps: List[SimulationStep] = []
    hops_traversed: List[str] = [source]
    current_node = source
    current_ttl = packet.ttl
    status = "IN_TRANSIT"
    summary = ""

    # Compile routing tables for all nodes according to mode
    def get_table(node: str) -> RoutingTable:
        if routing_mode == "weighted":
            return build_weighted_routing_table(topology, node)
        return build_routing_table(topology, node)

    # Edge case: Source is already destination
    if source == destination:
        steps.append(
            SimulationStep(
                hop_number=1,
                current_node=source,
                incoming_ttl=current_ttl,
                outgoing_ttl=current_ttl,
                action="TERMINAL",
                next_hop=None,
                status="DELIVERED",
                detail=f"Source is the destination ({source}). Immediate delivery.",
            )
        )
        return SimulationResult(
            packet_id=packet.packet_id,
            source=source,
            destination=destination,
            payload=payload,
            initial_ttl=initial_ttl,
            final_ttl=current_ttl,
            routing_mode=routing_mode,
            status="DELIVERED",
            hops_traversed=[source],
            steps=steps,
            delivered=True,
            summary=f"Packet delivered immediately at origin {source}.",
        )

    # Pre-check endpoint map: map (host, port) to node_id
    endpoint_to_node: Dict[Tuple[str, int], str] = {
        topology.get_endpoint(n): n for n in topology.nodes()
    }

    hop_count = 0
    while hop_count < max_hops:
        hop_count += 1

        # Check if we arrived at destination
        if current_node == destination:
            steps.append(
                SimulationStep(
                    hop_number=hop_count,
                    current_node=current_node,
                    incoming_ttl=current_ttl,
                    outgoing_ttl=current_ttl,
                    action="DELIVERY",
                    next_hop=None,
                    status="DELIVERED",
                    detail=f"Packet successfully received by destination endpoint {current_node}.",
                )
            )
            status = "DELIVERED"
            summary = (
                f"Packet successfully delivered to {destination} in {len(hops_traversed) - 1} "
                f"hop(s). Final TTL: {current_ttl}."
            )
            break

        # Check TTL expiry before forwarding
        if current_ttl <= 1:
            steps.append(
                SimulationStep(
                    hop_number=hop_count,
                    current_node=current_node,
                    incoming_ttl=current_ttl,
                    outgoing_ttl=0,
                    action="DROP",
                    next_hop=None,
                    status="DROPPED_TTL",
                    detail=f"TTL expired (TTL={current_ttl}) at node {current_node}. Packet dropped.",
                )
            )
            current_ttl = 0
            status = "DROPPED_TTL"
            summary = f"Packet dropped at {current_node} due to TTL expiry."
            break

        # Consult routing table at current_node
        rt = get_table(current_node)
        route_endpoint = rt.get_route(destination)

        if route_endpoint is None:
            steps.append(
                SimulationStep(
                    hop_number=hop_count,
                    current_node=current_node,
                    incoming_ttl=current_ttl,
                    outgoing_ttl=current_ttl,
                    action="DROP",
                    next_hop=None,
                    status="NO_ROUTE",
                    detail=f"Node {current_node} has no routing table entry for destination {destination}.",
                )
            )
            status = "NO_ROUTE"
            summary = f"Packet dropped at {current_node}: No route found for destination '{destination}'."
            break

        next_hop_node = endpoint_to_node.get(route_endpoint)
        if next_hop_node is None:
            # Fallback if endpoint unmapped
            next_hop_node = f"{route_endpoint[0]}:{route_endpoint[1]}"

        # Decrement TTL and forward
        outgoing_ttl = current_ttl - 1
        steps.append(
            SimulationStep(
                hop_number=hop_count,
                current_node=current_node,
                incoming_ttl=current_ttl,
                outgoing_ttl=outgoing_ttl,
                action="FORWARD",
                next_hop=next_hop_node,
                status="FORWARDING",
                detail=(
                    f"Forwarding from {current_node} to {next_hop_node} via {route_endpoint[0]}:{route_endpoint[1]}. "
                    f"TTL decremented: {current_ttl} -> {outgoing_ttl}."
                ),
            )
        )

        current_ttl = outgoing_ttl
        current_node = next_hop_node
        hops_traversed.append(current_node)

    else:
        # Loop limit exceeded
        status = "DROPPED_HOPS_EXCEEDED"
        summary = f"Forwarding exceeded maximum hop limit ({max_hops}). Possible forwarding cycle."

    return SimulationResult(
        packet_id=packet.packet_id,
        source=source,
        destination=destination,
        payload=payload,
        initial_ttl=initial_ttl,
        final_ttl=current_ttl,
        routing_mode=routing_mode,
        status=status,
        hops_traversed=hops_traversed,
        steps=steps,
        delivered=(status == "DELIVERED"),
        summary=summary,
    )


def generate_graphviz_dot(
    topology: NetworkTopology,
    highlight_path: Optional[List[str]] = None,
    failed_links: Optional[Set[Tuple[str, str]]] = None,
) -> str:
    """
    Generates a Graphviz DOT script rendering the topology graph.

    Visual encoding:
        - Routers: Rounded boxes, deep navy fill (#1E293B), bright cyan border (#38BDF8).
        - Receivers: Ellipses, dark emerald fill (#064E3B), light mint border (#34D399).
        - Normal Edges: Slate (#64748B), weight as clean label.
        - Highlighted Path Edges: Bold gold/amber stroke (#F59E0B), penwidth=3.5.
        - Failed Links: Dashed crimson stroke (#EF4444), label 'FAIL' or 'SEVERED'.
    """
    highlight_path = highlight_path or []
    failed_links = failed_links or set()

    # Normalize failed links to ordered tuples
    normalized_failed: Set[Tuple[str, str]] = {
        (min(u, v), max(u, v)) for u, v in failed_links
    }

    # Build set of highlighted undirected edge pairs
    path_edges: Set[Tuple[str, str]] = set()
    for i in range(len(highlight_path) - 1):
        u, v = highlight_path[i], highlight_path[i + 1]
        path_edges.add((min(u, v), max(u, v)))

    dot_lines = [
        'graph NetworkTopology {',
        '    bgcolor="transparent";',
        '    rankdir=LR;',
        '    node [fontname="Helvetica, Arial, sans-serif", fontsize=11, fontcolor="#F8FAFC"];',
        '    edge [fontname="Helvetica, Arial, sans-serif", fontsize=10, fontcolor="#94A3B8"];',
    ]

    nodes = topology.nodes()
    for node in sorted(nodes):
        is_receiver = node.startswith("Dest-") or node.startswith("receiver")
        is_in_path = node in highlight_path

        if is_receiver:
            border = "#F59E0B" if is_in_path else "#34D399"
            fill = "#78350F" if is_in_path else "#064E3B"
            dot_lines.append(
                f'    "{html.escape(node)}" ['
                f'shape=ellipse, style="filled", '
                f'fillcolor="{fill}", color="{border}", penwidth=2.0, '
                f'label="{html.escape(node)}\\n(Receiver)"];'
            )
        else:
            border = "#F59E0B" if is_in_path else "#38BDF8"
            fill = "#78350F" if is_in_path else "#1E293B"
            dot_lines.append(
                f'    "{html.escape(node)}" ['
                f'shape=box, style="filled,rounded", '
                f'fillcolor="{fill}", color="{border}", penwidth=2.0, '
                f'label="{html.escape(node)}\\n(Router)"];'
            )

    # Add active edges
    seen_edges: Set[Tuple[str, str]] = set()
    for u in sorted(nodes):
        for v in sorted(topology.neighbors(u)):
            pair = (min(u, v), max(u, v))
            if pair in seen_edges:
                continue
            seen_edges.add(pair)

            weight = topology.get_link_weight(u, v)
            is_failed = pair in normalized_failed
            is_highlighted = pair in path_edges and not is_failed

            if is_failed:
                dot_lines.append(
                    f'    "{html.escape(pair[0])}" -- "{html.escape(pair[1])}" ['
                    f'label=" {weight} (SEVERED) ", style="dashed", '
                    f'color="#EF4444", penwidth=2.5, fontcolor="#EF4444"];'
                )
            elif is_highlighted:
                dot_lines.append(
                    f'    "{html.escape(pair[0])}" -- "{html.escape(pair[1])}" ['
                    f'label=" {weight} ", '
                    f'color="#F59E0B", penwidth=3.5, fontcolor="#FBBF24", weight=2];'
                )
            else:
                dot_lines.append(
                    f'    "{html.escape(pair[0])}" -- "{html.escape(pair[1])}" ['
                    f'label=" {weight} ", '
                    f'color="#64748B", penwidth=1.5, fontcolor="#94A3B8"];'
                )

    # Render any failed links between registered nodes that were severed/removed from topology
    for pair in sorted(normalized_failed):
        if pair not in seen_edges and topology.has_node(pair[0]) and topology.has_node(pair[1]):
            dot_lines.append(
                f'    "{html.escape(pair[0])}" -- "{html.escape(pair[1])}" ['
                f'label=" SEVERED ", style="dashed", '
                f'color="#EF4444", penwidth=2.5, fontcolor="#EF4444"];'
            )

    dot_lines.append('}')
    return '\n'.join(dot_lines)
