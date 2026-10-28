"""
app.py - Interactive Streamlit Dashboard for DSA Network Packet Routing.

Demonstrates fundamental Data Structures & Algorithms (DSA) applied to network routing:
- Graph representations (Adjacency list with weighted edges)
- Breadth-First Search (BFS) for shortest-hop pathfinding
- Dijkstra's Algorithm for least-cost pathfinding
- Hop-by-hop packet forwarding with TTL decrement and framing semantics
- Dynamic network topology resilience and transactional route recalculation
"""

from __future__ import annotations

import copy
import html
from typing import Any, Dict, List, Optional, Set, Tuple

import streamlit as st

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
from NetworkPacketRouting.graph import DEFAULT_WEIGHT, InvalidWeightError
from NetworkPacketRouting.resilience import ResilienceError, TopologyRoutingManager
from NetworkPacketRouting.route_planner import (
    build_routing_table,
    build_weighted_routing_table,
)
from NetworkPacketRouting.router import Router
from NetworkPacketRouting.topology import (
    LinkError,
    LinkNotFoundError,
    NetworkTopology,
    NodeNotFoundError,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DSA Network Packet Routing Simulator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (High-contrast, professional engineering theme)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre, [class*="stCode"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 1.25rem 1.75rem;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .main-title {
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
    }
    .main-subtitle {
        color: #94A3B8;
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.4;
    }

    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        text-align: center;
        transition: border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #38BDF8;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.15rem;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
    }

    .legend-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1.2rem;
        align-items: center;
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #E2E8F0;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .legend-badge {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 3px;
    }

    .algo-panel {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        height: 100%;
    }
    .algo-panel-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .algo-stat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #1E293B;
        font-size: 0.9rem;
        color: #E2E8F0;
    }
    .algo-stat-label {
        color: #94A3B8;
        font-weight: 500;
    }
    .algo-stat-value {
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    .insight-card {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #F8FAFC;
        font-size: 0.92rem;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .insight-card-alert {
        background-color: #0F172A;
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #F8FAFC;
        font-size: 0.92rem;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    .sim-status-banner {
        border-radius: 8px;
        padding: 0.85rem 1.25rem;
        margin: 1rem 0;
        font-weight: 600;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .sim-delivered {
        background-color: #064E3B;
        color: #A7F3D0;
        border: 1px solid #059669;
    }
    .sim-dropped {
        background-color: #7F1D1D;
        color: #FECACA;
        border: 1px solid #DC2626;
    }
    .sim-warning {
        background-color: #78350F;
        color: #FDE68A;
        border: 1px solid #D97706;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization & Resilience
# -----------------------------------------------------------------------------
if (
    "topology" not in st.session_state
    or not isinstance(st.session_state.get("topology"), NetworkTopology)
):
    st.session_state["topology"] = create_demo_topology()
if "failed_links" not in st.session_state:
    st.session_state["failed_links"] = set()
if "severed_links" not in st.session_state:
    st.session_state["severed_links"] = {}
if "selected_source" not in st.session_state:
    st.session_state["selected_source"] = "R1"
if "selected_destination" not in st.session_state:
    st.session_state["selected_destination"] = "Dest-Alpha"
if "routing_mode" not in st.session_state:
    st.session_state["routing_mode"] = "weighted"  # "weighted" or "unweighted"
if "highlight_mode" not in st.session_state:
    st.session_state["highlight_mode"] = "active"  # "active", "dijkstra", "bfs", "none"
if "last_simulation" not in st.session_state:
    st.session_state["last_simulation"] = None
if "sim_ttl" not in st.session_state:
    st.session_state["sim_ttl"] = 8
if "sim_payload" not in st.session_state:
    st.session_state["sim_payload"] = "DSA Packet Transmission"

topo: NetworkTopology = st.session_state["topology"]


def reset_topology() -> None:
    """Resets session topology to the default demo diamond topology."""
    st.session_state["topology"] = create_demo_topology()
    st.session_state["failed_links"] = set()
    st.session_state["severed_links"] = {}
    st.session_state["last_simulation"] = None
    st.session_state["selected_source"] = "R1"
    st.session_state["selected_destination"] = "Dest-Alpha"
    st.session_state["routing_mode"] = "weighted"
    st.session_state["highlight_mode"] = "active"


# -----------------------------------------------------------------------------
# Header Component
# -----------------------------------------------------------------------------
metrics = get_topology_metrics(topo)

st.markdown(
    """
    <div class="main-header">
        <h1 class="main-title">Network Packet Routing & Graph DSA Simulator</h1>
        <p class="main-subtitle">
            An interactive visual testbed comparing <b>Breadth-First Search (BFS)</b> shortest-hop routing
            against <b>Dijkstra's Algorithm</b> least-cost routing across dynamic, fault-resilient network topologies.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Top Metric Bar
# -----------------------------------------------------------------------------
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
with m_col1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{metrics["node_count"]}</div><div class="metric-label">Nodes Total</div></div>',
        unsafe_allow_html=True,
    )
with m_col2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{metrics["router_count"]}</div><div class="metric-label">Active Routers</div></div>',
        unsafe_allow_html=True,
    )
with m_col3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{metrics["receiver_count"]}</div><div class="metric-label">Receivers (Leafs)</div></div>',
        unsafe_allow_html=True,
    )
with m_col4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{metrics["link_count"]}</div><div class="metric-label">Active Links</div></div>',
        unsafe_allow_html=True,
    )
with m_col5:
    failed_count = len(st.session_state["failed_links"])
    status_color = "#34D399" if failed_count == 0 else "#EF4444"
    st.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color: {status_color};">{failed_count}</div><div class="metric-label">Severed Links</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Node Lists & State Consistency Helpers
# -----------------------------------------------------------------------------
all_nodes = sorted(topo.nodes())
routers = [n for n in all_nodes if not n.startswith("Dest-") and not n.startswith("receiver")]
receivers = [n for n in all_nodes if n.startswith("Dest-") or n.startswith("receiver")]

# Ensure active selections exist in current topology
if st.session_state["selected_source"] not in routers:
    st.session_state["selected_source"] = routers[0] if routers else "R1"
dest_options = [n for n in all_nodes if n != st.session_state["selected_source"]]
if st.session_state["selected_destination"] not in dest_options:
    st.session_state["selected_destination"] = dest_options[0] if dest_options else "Dest-Alpha"


# -----------------------------------------------------------------------------
# Sidebar: Global Navigation & Scenario Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls & Parameters")

    if st.button("Reset to Default Topology", use_container_width=True):
        reset_topology()
        st.rerun()

    st.markdown("---")
    st.subheader("Route Query Selection")

    # Source Selection
    curr_src = st.session_state["selected_source"]
    src_idx = routers.index(curr_src) if curr_src in routers else 0

    def on_sidebar_source_change():
        st.session_state["selected_source"] = st.session_state["sb_source"]
        # Invalidate previous simulation when source changes
        st.session_state["last_simulation"] = None

    selected_source = st.selectbox(
        "Source Router",
        options=routers,
        index=src_idx,
        key="sb_source",
        on_change=on_sidebar_source_change,
        help="Select origin router for path computation and packet simulation.",
    )

    # Recompute destination options based on selected source
    valid_dests = [n for n in all_nodes if n != st.session_state["selected_source"]]
    curr_dest = st.session_state["selected_destination"]
    dest_idx = valid_dests.index(curr_dest) if curr_dest in valid_dests else 0

    def on_sidebar_dest_change():
        st.session_state["selected_destination"] = st.session_state["sb_dest"]
        # Invalidate previous simulation when destination changes
        st.session_state["last_simulation"] = None

    selected_dest = st.selectbox(
        "Destination Node",
        options=valid_dests,
        index=dest_idx,
        key="sb_dest",
        on_change=on_sidebar_dest_change,
        help="Target node for pathfinding (can be another router or receiver endpoint).",
    )

    st.markdown("---")
    st.subheader("Routing Strategy")

    mode_options = ["Weighted (Dijkstra - Least Cost)", "Unweighted (BFS - Fewest Hops)"]
    curr_mode_idx = 0 if st.session_state["routing_mode"] == "weighted" else 1

    def on_mode_change():
        chosen = st.session_state["sb_mode"]
        st.session_state["routing_mode"] = "weighted" if "Weighted" in chosen else "unweighted"
        st.session_state["last_simulation"] = None

    selected_mode_str = st.selectbox(
        "Active Forwarding Strategy",
        options=mode_options,
        index=curr_mode_idx,
        key="sb_mode",
        on_change=on_mode_change,
        help="Determines which routing table compiler generates the forwarding path.",
    )

    st.markdown("---")
    st.subheader("Graph Path Highlight")
    highlight_options = [
        "Active Strategy Route",
        "Dijkstra Path (Least Cost)",
        "BFS Path (Fewest Hops)",
        "None",
    ]
    curr_hl = st.session_state["highlight_mode"]
    hl_idx = 0 if curr_hl == "active" else (1 if curr_hl == "dijkstra" else (2 if curr_hl == "bfs" else 3))
    selected_hl = st.radio(
        "Highlight Route in Topology:",
        options=highlight_options,
        index=hl_idx,
    )
    if "Active" in selected_hl:
        st.session_state["highlight_mode"] = "active"
    elif "Dijkstra" in selected_hl:
        st.session_state["highlight_mode"] = "dijkstra"
    elif "BFS" in selected_hl:
        st.session_state["highlight_mode"] = "bfs"
    else:
        st.session_state["highlight_mode"] = "none"

    st.markdown("---")
    st.caption("DSA Network Packet Routing Simulator · Production Engine")


# -----------------------------------------------------------------------------
# Compute Route Comparison for Current Source & Destination
# -----------------------------------------------------------------------------
source_node = st.session_state["selected_source"]
destination_node = st.session_state["selected_destination"]
active_mode = st.session_state["routing_mode"]

try:
    comparison = compare_routes(topo, source_node, destination_node)
except Exception as exc:
    comparison = None
    st.error(f"Error computing paths from {source_node} to {destination_node}: {exc}")

# Determine path to highlight in Graphviz visualization
active_highlight_path: List[str] = []
if comparison:
    hl_pref = st.session_state["highlight_mode"]
    if hl_pref == "active":
        active_highlight_path = (
            comparison.dijkstra_path if active_mode == "weighted" else comparison.bfs_path
        ) or []
    elif hl_pref == "dijkstra" and comparison.dijkstra_path:
        active_highlight_path = comparison.dijkstra_path
    elif hl_pref == "bfs" and comparison.bfs_path:
        active_highlight_path = comparison.bfs_path


# -----------------------------------------------------------------------------
# Main Navigation Tabs (Clean 4 Sections - No Emojis in Tab Labels)
# -----------------------------------------------------------------------------
tab_topo, tab_compare, tab_sim, tab_resilience = st.tabs([
    "Network Topology",
    "Route Comparison",
    "Packet Simulator",
    "Resilience & Link Management",
])


# =============================================================================
# TAB 1: Network Topology Visualizer
# =============================================================================
with tab_topo:
    st.subheader("Network Topology Graph")
    st.caption(
        f"Active journey: **{source_node} ➔ {destination_node}** using "
        f"**{'Weighted Dijkstra' if active_mode == 'weighted' else 'Unweighted BFS'}** routing."
    )

    dot_code = generate_graphviz_dot(
        topo,
        highlight_path=active_highlight_path,
        failed_links=st.session_state["failed_links"],
    )
    st.graphviz_chart(dot_code, use_container_width=True)

    # Clean Legend Bar
    active_path_str = " ➔ ".join(active_highlight_path) if active_highlight_path else "None"
    st.markdown(
        f"""
        <div class="legend-container">
            <div class="legend-item">
                <span class="legend-badge" style="background-color: #1E293B; border: 1.5px solid #38BDF8;"></span>
                <span><b>Core Router</b> (Box)</span>
            </div>
            <div class="legend-item">
                <span class="legend-badge" style="background-color: #064E3B; border: 1.5px solid #34D399; border-radius: 50%;"></span>
                <span><b>Receiver Endpoint</b> (Ellipse)</span>
            </div>
            <div class="legend-item">
                <span class="legend-badge" style="background-color: #F59E0B;"></span>
                <span><b>Highlighted Path:</b> <code>{active_path_str}</code></span>
            </div>
            <div class="legend-item">
                <span class="legend-badge" style="background-color: #EF4444; border-top: 2px dashed #EF4444;"></span>
                <span><b>Severed Link</b> (Dashed Red)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compact Secondary Data Tables
    t_c1, t_c2 = st.columns(2)
    with t_c1:
        st.markdown("#### Registered Endpoints & Roles")
        node_records = []
        for n in sorted(topo.nodes()):
            host, port = topo.get_endpoint(n)
            is_rec = n.startswith("Dest-") or n.startswith("receiver")
            role = "Receiver Endpoint" if is_rec else "Core Router"
            node_records.append({
                "Node ID": n,
                "Role": role,
                "Endpoint Address": f"{host}:{port}",
                "Links": len(topo.neighbors(n)),
            })
        st.dataframe(node_records, hide_index=True, use_container_width=True)

    with t_c2:
        st.markdown("#### Link Weights & Status")
        link_records = []
        for u, v, w in sorted(metrics["links"], key=lambda x: (x[0], x[1])):
            is_sev = (min(u, v), max(u, v)) in {
                (min(a, b), max(a, b)) for a, b in st.session_state["failed_links"]
            }
            status_str = "SEVERED" if is_sev else "ACTIVE"
            link_records.append({
                "Link": f"{u} ⟷ {v}",
                "Weight (Latency/Cost)": w,
                "Status": status_str,
            })
        st.dataframe(link_records, hide_index=True, use_container_width=True)


# =============================================================================
# TAB 2: Route Comparison (BFS vs Dijkstra)
# =============================================================================
with tab_compare:
    st.subheader(f"Route Comparison: {source_node} ➔ {destination_node}")
    st.caption(
        "Demonstrates algorithmic trade-offs between unweighted shortest-hop routing (BFS) "
        "and weighted least-cost pathfinding (Dijkstra)."
    )

    if comparison:
        c1, c2 = st.columns(2)

        # BFS Panel
        with c1:
            st.markdown(
                """
                <div class="algo-panel">
                    <div class="algo-panel-title" style="color: #60A5FA;">
                        <span>Breadth-First Search (BFS)</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            if comparison.bfs_path:
                st.markdown(
                    f"""
                    <div class="algo-stat-row"><span class="algo-stat-label">Optimization Goal:</span><span class="algo-stat-value">Fewest Hops (Unweighted)</span></div>
                    <div class="algo-stat-row"><span class="algo-stat-label">Computed Path:</span><span class="algo-stat-value">{' ➔ '.join(comparison.bfs_path)}</span></div>
                    <div class="algo-stat-row"><span class="algo-stat-label">Total Hop Count:</span><span class="algo-stat-value">{comparison.bfs_hops} hop(s)</span></div>
                    <div class="algo-stat-row"><span class="algo-stat-label">Total Path Cost:</span><span class="algo-stat-value">{comparison.bfs_cost}</span></div>
                    <div class="algo-stat-row"><span class="algo-stat-label">First Forwarding Hop:</span><span class="algo-stat-value">{comparison.bfs_next_hop}</span></div>
                    """,
                    unsafe_allow_html=True,
                )
                bfs_rt = build_routing_table(topo, source_node)
                ep = bfs_rt.get_route(destination_node)
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                st.code(f"RoutingTable[{source_node}].get_route('{destination_node}') -> {ep}", language="python")
            else:
                st.warning(f"Destination {destination_node} is unreachable from {source_node} via BFS.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Dijkstra Panel
        with c2:
            st.markdown(
                """
                <div class="algo-panel">
                    <div class="algo-panel-title" style="color: #34D399;">
                        <span>Dijkstra's Algorithm</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            if comparison.dijkstra_path:
                st.markdown(
                    f"""
                    <div class="algo-stat-row"><span class="algo-stat-label">Optimization Goal:</span><span class="algo-stat-value">Minimum Cost / Latency</span></div>
                    <div class="algo-stat-row"><span class="algo-stat-label">Computed Path:</span><span class="algo-stat-value">{' ➔ '.join(comparison.dijkstra_path)}</span></div>
                    <div class="algo-stat-row"><span class="algo-stat-label">Total Hop Count:</span><span class="algo-stat-value">{comparison.dijkstra_hops} hop(s)</span></div>
                    <div class="algo-stat-row"><span class="algo-stat-label">Total Path Cost:</span><span class="algo-stat-value">{comparison.dijkstra_cost}</span></div>
                    <div class="algo-stat-row"><span class="algo-stat-label">First Forwarding Hop:</span><span class="algo-stat-value">{comparison.dijkstra_next_hop}</span></div>
                    """,
                    unsafe_allow_html=True,
                )
                dijkstra_rt = build_weighted_routing_table(topo, source_node)
                ep = dijkstra_rt.get_route(destination_node)
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                st.code(f"RoutingTable[{source_node}].get_route('{destination_node}') -> {ep}", language="python")
            else:
                st.warning(f"Destination {destination_node} is unreachable from {source_node} via Dijkstra.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Algorithmic Analysis Card
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if comparison.is_different_path and comparison.is_cost_divergence:
            st.markdown(
                f"""
                <div class="insight-card-alert">
                    <b>Core Algorithmic Trade-off Observed:</b><br>
                    • <b>BFS</b> chose a <b>{comparison.bfs_hops}-hop path</b> with a high weight of <b>{comparison.bfs_cost}</b> because it optimizes strictly for fewest edge traversals (O(V + E)).<br>
                    • <b>Dijkstra</b> chose a <b>{comparison.dijkstra_hops}-hop path</b> with a significantly lower total weight of <b>{comparison.dijkstra_cost}</b> using its min-heap priority queue (O((V + E) log V)).<br>
                    <b>Practical Impact:</b> In production networks, routing solely by hop count can route traffic across expensive or slow WAN/satellite links, causing severe congestion.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif not comparison.is_different_path and comparison.bfs_path:
            st.markdown(
                f"""
                <div class="insight-card">
                    <b>Algorithmic Convergence:</b> Both BFS and Dijkstra selected the identical optimal route:
                    <code>{' ➔ '.join(comparison.dijkstra_path)}</code> (Hops: {comparison.dijkstra_hops}, Cost: {comparison.dijkstra_cost}).
                    This occurs because the fewest-hop path also happens to be the path of lowest total weight.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error(f"Destination {destination_node} is disconnected from source {source_node}.")

        # Expandable Routing Table Inspection
        with st.expander(f"Inspect Complete Forwarding Tables for Router {source_node}"):
            rt_c1, rt_c2 = st.columns(2)
            with rt_c1:
                st.markdown("**BFS Routing Table (Fewest Hops)**")
                bfs_table = build_routing_table(topo, source_node)
                b_entries = []
                for dest in sorted(topo.nodes()):
                    if dest != source_node:
                        ep = bfs_table.get_route(dest)
                        b_entries.append({"Destination": dest, "Next Hop Address": f"{ep[0]}:{ep[1]}" if ep else "NO ROUTE"})
                st.dataframe(b_entries, hide_index=True, use_container_width=True)

            with rt_c2:
                st.markdown("**Dijkstra Routing Table (Least Cost)**")
                dijkstra_table = build_weighted_routing_table(topo, source_node)
                d_entries = []
                for dest in sorted(topo.nodes()):
                    if dest != source_node:
                        ep = dijkstra_table.get_route(dest)
                        d_entries.append({"Destination": dest, "Next Hop Address": f"{ep[0]}:{ep[1]}" if ep else "NO ROUTE"})
                st.dataframe(d_entries, hide_index=True, use_container_width=True)


# =============================================================================
# TAB 3: Packet Simulator
# =============================================================================
with tab_sim:
    st.subheader("In-Process Packet Forwarding Simulation")

    st.info(
        "ℹ️ **In-Process Simulation Model**: Simulates packet forwarding in-process "
        "using the project's exact `Packet` dataclass and compiled `RoutingTable` lookups. "
        "TTL is validated and decremented at each router hop. "
        "For concurrent multi-threaded TCP socket forwarding over physical network ports, "
        "run the headless test suite (`test_topology_router.py`)."
    )

    st.markdown("#### Transmission Configuration")
    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns([2, 2, 2, 2])

    with sim_c1:
        # Keep source selection synchronized
        sim_src_idx = routers.index(source_node) if source_node in routers else 0

        def on_sim_src_change():
            st.session_state["selected_source"] = st.session_state["in_sim_src"]
            st.session_state["last_simulation"] = None

        sim_source = st.selectbox(
            "Origin Router (src)",
            options=routers,
            index=sim_src_idx,
            key="in_sim_src",
            on_change=on_sim_src_change,
        )

    with sim_c2:
        # Keep destination selection synchronized
        sim_dest_opts = [n for n in all_nodes if n != st.session_state["selected_source"]]
        sim_dest_idx = sim_dest_opts.index(destination_node) if destination_node in sim_dest_opts else 0

        def on_sim_dest_change():
            st.session_state["selected_destination"] = st.session_state["in_sim_dest"]
            st.session_state["last_simulation"] = None

        sim_destination = st.selectbox(
            "Target Node (dest)",
            options=sim_dest_opts,
            index=sim_dest_idx,
            key="in_sim_dest",
            on_change=on_sim_dest_change,
        )

    with sim_c3:
        sim_ttl = st.slider(
            "Initial TTL",
            min_value=1,
            max_value=15,
            value=st.session_state["sim_ttl"],
            step=1,
            help="Packet Time-To-Live. Decremented by 1 at each router hop.",
        )
        st.session_state["sim_ttl"] = sim_ttl

    with sim_c4:
        mode_select_opts = ["Weighted (Dijkstra)", "Unweighted (BFS)"]
        sim_mode_idx = 0 if active_mode == "weighted" else 1

        def on_sim_mode_change():
            chosen = st.session_state["in_sim_mode"]
            st.session_state["routing_mode"] = "weighted" if "Weighted" in chosen else "unweighted"
            st.session_state["last_simulation"] = None

        sim_mode_choice = st.selectbox(
            "Routing Strategy",
            options=mode_select_opts,
            index=sim_mode_idx,
            key="in_sim_mode",
            on_change=on_sim_mode_change,
        )

    sim_payload = st.text_input(
        "Packet Payload",
        value=st.session_state["sim_payload"],
        help="Arbitrary user string payload wrapped in the framed packet.",
    )
    st.session_state["sim_payload"] = sim_payload

    transmit_pressed = st.button("Transmit Packet", type="primary", use_container_width=True)

    # Trigger simulation on button press
    if transmit_pressed:
        strategy_key = "weighted" if "Weighted" in sim_mode_choice else "unweighted"
        result = simulate_packet_forwarding(
            topo,
            source=st.session_state["selected_source"],
            destination=st.session_state["selected_destination"],
            payload=sim_payload,
            initial_ttl=sim_ttl,
            routing_mode=strategy_key,
        )
        st.session_state["last_simulation"] = result

    # Display simulation trace
    sim_result: Optional[SimulationResult] = st.session_state.get("last_simulation")

    # If simulation result exists but belongs to an outdated source or destination, recompute
    if sim_result is not None:
        active_src = st.session_state["selected_source"]
        active_dest = st.session_state["selected_destination"]
        active_strategy = "weighted" if "Weighted" in sim_mode_choice else "unweighted"
        if (
            sim_result.source != active_src
            or sim_result.destination != active_dest
            or sim_result.routing_mode != active_strategy
        ):
            sim_result = simulate_packet_forwarding(
                topo,
                source=active_src,
                destination=active_dest,
                payload=sim_payload,
                initial_ttl=sim_ttl,
                routing_mode=active_strategy,
            )
            st.session_state["last_simulation"] = sim_result

    if sim_result:
        st.markdown("---")
        st.markdown("### Packet Forwarding Trace")

        # Prominent Result Status Banner
        if sim_result.delivered:
            st.markdown(
                f'<div class="sim-status-banner sim-delivered">✅ {sim_result.summary}</div>',
                unsafe_allow_html=True,
            )
        elif sim_result.status == "DROPPED_TTL":
            st.markdown(
                f'<div class="sim-status-banner sim-dropped">🛑 {sim_result.summary} '
                f'(Initial TTL={sim_result.initial_ttl} was insufficient for path length)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="sim-status-banner sim-warning">⚠️ {sim_result.summary}</div>',
                unsafe_allow_html=True,
            )

        # Key Metrics Row
        s_m1, s_m2, s_m3, s_m4, s_m5 = st.columns(5)
        with s_m1:
            st.metric("Destination Node", sim_result.destination)
        with s_m2:
            st.metric("Final Status", sim_result.status)
        with s_m3:
            st.metric("Hops Traversed", len(sim_result.hops_traversed) - 1)
        with s_m4:
            st.metric("TTL (Start ➔ End)", f"{sim_result.initial_ttl} ➔ {sim_result.final_ttl}")
        with s_m5:
            st.metric("Traversal Path", " ➔ ".join(sim_result.hops_traversed))

        # Hop-by-Hop Decision Table
        st.markdown("#### Hop Decisions & TTL History")
        step_records = []
        for step in sim_result.steps:
            step_records.append({
                "Hop #": step.hop_number,
                "Node ID": step.current_node,
                "TTL In": step.incoming_ttl,
                "TTL Out": step.outgoing_ttl,
                "Action": step.action,
                "Next Hop": step.next_hop if step.next_hop else "—",
                "Outcome": step.status,
                "Forwarding Detail": step.detail,
            })
        st.dataframe(step_records, hide_index=True, use_container_width=True)
    else:
        st.markdown("---")
        st.caption("Click **Transmit Packet** to execute an in-process forwarding trace.")


# =============================================================================
# TAB 4: Resilience & Link Management
# =============================================================================
with tab_resilience:
    st.subheader("Dynamic Network Resilience & Fault Injection")
    st.caption(
        "Demonstrates dynamic link failure, restoration, and weight adjustment. "
        "Routing tables recalculate automatically to maintain optimal forwarding paths."
    )

    r_col1, r_col2, r_col3 = st.columns(3)

    # 1. Sever Link
    with r_col1:
        st.markdown("#### Sever Link")
        st.caption("Simulate link failure or physical link cut:")
        existing_links = metrics["links"]
        link_options = [f"{u} ⟷ {v} (weight {w})" for u, v, w in existing_links]

        if link_options:
            selected_link_str = st.selectbox("Active Link to Sever:", options=link_options, key="sev_link_sel")
            link_parts = selected_link_str.split(" ⟷ ")
            u_sev = link_parts[0]
            v_sev = link_parts[1].split(" ")[0]

            if st.button(f"Sever Link ({u_sev} ⟷ {v_sev})", type="secondary", use_container_width=True):
                try:
                    orig_weight = topo.get_link_weight(u_sev, v_sev)
                    topo.disconnect(u_sev, v_sev)
                    pair = (min(u_sev, v_sev), max(u_sev, v_sev))
                    st.session_state["failed_links"].add(pair)
                    st.session_state["severed_links"][pair] = orig_weight
                    st.session_state["last_simulation"] = None
                    st.success(f"Severed link {u_sev} ⟷ {v_sev}. Routes recalculated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to sever link: {exc}")
        else:
            st.info("No active links available to sever.")

    # 2. Restore Link
    with r_col2:
        st.markdown("#### Restore Link")
        st.caption("Restore previously severed link:")
        if st.session_state["failed_links"]:
            failed_opts = []
            for u, v in sorted(st.session_state["failed_links"]):
                w = st.session_state["severed_links"].get((u, v), 1.0)
                failed_opts.append(f"{u} ⟷ {v} (saved weight {w})")

            restore_choice = st.selectbox("Severed Link to Restore:", options=failed_opts, key="res_link_sel")
            if st.button("Restore Link", use_container_width=True):
                r_parts = restore_choice.split(" ⟷ ")
                r_u = r_parts[0]
                r_v = r_parts[1].split(" ")[0]
                pair = (min(r_u, r_v), max(r_u, r_v))
                saved_w = st.session_state["severed_links"].get(pair, 1.0)
                try:
                    topo.connect(r_u, r_v, weight=saved_w)
                    st.session_state["failed_links"].discard(pair)
                    st.session_state["severed_links"].pop(pair, None)
                    st.session_state["last_simulation"] = None
                    st.success(f"Restored link {r_u} ⟷ {r_v} with weight {saved_w}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to restore link: {exc}")
        else:
            st.info("No severed links. All topology links are active.")

    # 3. Add or Update Link Weight
    with r_col3:
        st.markdown("#### Update / Add Link")
        st.caption("Change link latency or connect new nodes:")
        ep_a = st.selectbox("Endpoint A", options=all_nodes, index=0, key="edit_ep_a")
        ep_b_opts = [n for n in all_nodes if n != ep_a]
        ep_b = st.selectbox("Endpoint B", options=ep_b_opts, index=0, key="edit_ep_b")
        curr_w = 1.0
        if topo.has_link(ep_a, ep_b):
            curr_w = float(topo.get_link_weight(ep_a, ep_b))
        new_weight = st.number_input(
            "Weight (Latency / Cost)",
            min_value=0.0,
            max_value=1000.0,
            value=curr_w,
            step=1.0,
            key="edit_w_input",
        )

        if st.button("Apply Link Weight / Connect", use_container_width=True):
            try:
                pair = (min(ep_a, ep_b), max(ep_a, ep_b))
                if topo.has_link(ep_a, ep_b):
                    topo.set_link_weight(ep_a, ep_b, new_weight)
                    st.success(f"Updated link weight for {ep_a} ⟷ {ep_b} to {new_weight}.")
                else:
                    topo.connect(ep_a, ep_b, weight=new_weight)
                    st.success(f"Connected link {ep_a} ⟷ {ep_b} with weight {new_weight}.")
                st.session_state["failed_links"].discard(pair)
                st.session_state["severed_links"].pop(pair, None)
                st.session_state["last_simulation"] = None
                st.rerun()
            except Exception as exc:
                st.error(f"Operation failed: {exc}")

    # Active Route Impact Summary
    st.markdown("---")
    st.markdown(f"#### Live Route Status: {source_node} ➔ {destination_node}")
    if comparison and comparison.dijkstra_path:
        st.markdown(
            f"• **Dijkstra Least-Cost Path:** `{' ➔ '.join(comparison.dijkstra_path)}` "
            f"(Hops: {comparison.dijkstra_hops}, Total Weight: {comparison.dijkstra_cost})<br>"
            f"• **BFS Fewest-Hop Path:** `{' ➔ '.join(comparison.bfs_path) if comparison.bfs_path else 'None'}` "
            f"(Hops: {comparison.bfs_hops}, Total Weight: {comparison.bfs_cost})",
            unsafe_allow_html=True,
        )
    else:
        st.error(f"⚠️ Destination {destination_node} is currently unreachable from {source_node}.")

    # Two-Phase Commit Transaction Verification
    with st.expander("TopologyRoutingManager Two-Phase Commit Guarantee"):
        st.caption(
            "The `TopologyRoutingManager` enforces fault isolation: candidate forwarding tables "
            "are calculated on a deep copy of the topology before mutating live routers. "
            "If an invariant fails, changes roll back without disrupting active traffic."
        )
        mgr = TopologyRoutingManager(topo, default_routing_mode="weighted")
        mgr_records = []
        for r_id in routers:
            dummy_router = Router("127.0.0.1", topo.get_endpoint(r_id)[1])
            mgr.add_router(r_id, dummy_router, routing_mode="weighted")
            mgr_records.append({
                "Router ID": r_id,
                "Managed Strategy": mgr.get_routing_mode(r_id),
                "Synchronized Routes": len(mgr.get_router(r_id).routing_table),
            })
        st.dataframe(mgr_records, hide_index=True, use_container_width=True)
        st.success("All managed router tables verified with 100% transactional consistency.")
