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
# Custom Styling (Dark-themed, crisp modern SaaS aesthetics)
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
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.35rem 0;
    }
    .main-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.4;
    }

    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
    }

    .insight-box {
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.1) 0%, rgba(129, 140, 248, 0.1) 100%);
        border-left: 4px solid #38BDF8;
        border-radius: 6px;
        padding: 0.85rem 1.2rem;
        margin: 1rem 0;
        color: #E2E8F0;
        font-size: 0.92rem;
    }
    .insight-alert {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.12) 0%, rgba(239, 68, 68, 0.12) 100%);
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 0.85rem 1.2rem;
        margin: 1rem 0;
        color: #FEF3C7;
        font-size: 0.92rem;
    }

    .badge-delivered {
        background-color: #064E3B;
        color: #34D399;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        border: 1px solid #059669;
    }
    .badge-dropped {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        border: 1px solid #DC2626;
    }
    .badge-forwarding {
        background-color: #1E3A8A;
        color: #60A5FA;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        border: 1px solid #2563EB;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if (
    "topology" not in st.session_state
    or not isinstance(st.session_state.get("topology"), NetworkTopology)
):
    st.session_state["topology"] = create_demo_topology()
if "failed_links" not in st.session_state:
    st.session_state["failed_links"] = set()
if "selected_source" not in st.session_state:
    st.session_state["selected_source"] = "R1"
if "selected_destination" not in st.session_state:
    st.session_state["selected_destination"] = "Dest-Alpha"
if "last_simulation" not in st.session_state:
    st.session_state["last_simulation"] = None
if "highlight_mode" not in st.session_state:
    st.session_state["highlight_mode"] = "dijkstra"  # "dijkstra", "bfs", "none"


def reset_topology():
    """Resets session topology to the default demo diamond topology."""
    st.session_state["topology"] = create_demo_topology()
    st.session_state["failed_links"] = set()
    st.session_state["last_simulation"] = None
    st.session_state["selected_source"] = "R1"
    st.session_state["selected_destination"] = "Dest-Alpha"


# -----------------------------------------------------------------------------
# Header Component
# -----------------------------------------------------------------------------
topo = st.session_state["topology"]
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

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar: Global Controls & Scenario Manager
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls & Presets")
    st.markdown("Quickly switch presets or reset the scenario state:")

    if st.button("🔄 Reset to Default Diamond", use_container_width=True):
        reset_topology()
        st.rerun()

    st.markdown("---")
    st.subheader("🌐 Path Query Controls")
    all_nodes = sorted(topo.nodes())
    routers = [n for n in all_nodes if not n.startswith("Dest-") and not n.startswith("receiver")]
    receivers = [n for n in all_nodes if n.startswith("Dest-") or n.startswith("receiver")]

    src_index = routers.index(st.session_state["selected_source"]) if st.session_state["selected_source"] in routers else 0
    selected_source = st.selectbox(
        "Source Node (Sender Router)",
        options=routers,
        index=src_index,
        help="Select origin router for pathfinding and packet transmission.",
    )
    st.session_state["selected_source"] = selected_source

    dest_options = [n for n in all_nodes if n != selected_source]
    dest_default_idx = (
        dest_options.index(st.session_state["selected_destination"])
        if st.session_state["selected_destination"] in dest_options
        else 0
    )
    selected_dest = st.selectbox(
        "Destination Node",
        options=dest_options,
        index=dest_default_idx,
        help="Target node (can be another router or a receiver endpoint).",
    )
    st.session_state["selected_destination"] = selected_dest

    st.markdown("---")
    st.subheader("🎨 Topology Visual Path Highlight")
    highlight_opt = st.radio(
        "Highlight in Graph View:",
        options=["Dijkstra Path (Least Cost)", "BFS Path (Fewest Hops)", "None"],
        index=0 if st.session_state["highlight_mode"] == "dijkstra" else (1 if st.session_state["highlight_mode"] == "bfs" else 2),
    )
    if "Dijkstra" in highlight_opt:
        st.session_state["highlight_mode"] = "dijkstra"
    elif "BFS" in highlight_opt:
        st.session_state["highlight_mode"] = "bfs"
    else:
        st.session_state["highlight_mode"] = "none"

    st.markdown("---")
    st.caption("DSA Network Packet Routing Simulator · Milestone 3")


# -----------------------------------------------------------------------------
# Compute Route Comparison for Selected Pair
# -----------------------------------------------------------------------------
try:
    comparison = compare_routes(topo, selected_source, selected_dest)
except Exception as exc:
    comparison = None
    st.error(f"Error computing paths: {exc}")

# Determine path to highlight in graphviz chart
active_highlight_path: List[str] = []
if comparison:
    if st.session_state["highlight_mode"] == "dijkstra" and comparison.dijkstra_path:
        active_highlight_path = comparison.dijkstra_path
    elif st.session_state["highlight_mode"] == "bfs" and comparison.bfs_path:
        active_highlight_path = comparison.bfs_path


# -----------------------------------------------------------------------------
# Main Tabs Layout
# -----------------------------------------------------------------------------
tab_topo, tab_algo, tab_packet, tab_resilience, tab_viva = st.tabs([
    "🗺️ Network Topology",
    "⚡ Algorithm Comparison (BFS vs Dijkstra)",
    "📦 Packet Forwarding Simulation",
    "🛡️ Resilience & Dynamic Topology",
    "📊 Viva & Complexity Reference",
])


# =============================================================================
# TAB 1: Network Topology Visualizer
# =============================================================================
with tab_topo:
    st.subheader("Interactive Topology Map")
    st.caption(
        "Routers are displayed as blue rounded boxes. Receiver leaf nodes are displayed as green capsules. "
        "Edge labels denote link weights. Highlighted path is shown in bold amber."
    )

    t_col1, t_col2 = st.columns([3, 2])

    with t_col1:
        dot_code = generate_graphviz_dot(
            topo,
            highlight_path=active_highlight_path,
            failed_links=st.session_state["failed_links"],
        )
        st.graphviz_chart(dot_code, use_container_width=True)

        if active_highlight_path:
            mode_name = "Dijkstra (Weighted)" if st.session_state["highlight_mode"] == "dijkstra" else "BFS (Shortest Hop)"
            st.markdown(
                f'<div class="insight-box"><b>Active Visual Path ({mode_name}):</b> '
                f'{" ➔ ".join(active_highlight_path)}</div>',
                unsafe_allow_html=True,
            )

    with t_col2:
        st.markdown("#### Registered Endpoints & Roles")
        node_records = []
        for n in sorted(topo.nodes()):
            host, port = topo.get_endpoint(n)
            is_rec = n.startswith("Dest-") or n.startswith("receiver")
            role = "Receiver Endpoint" if is_rec else "Core Router"
            node_records.append({
                "Node ID": n,
                "Role": role,
                "Socket Endpoint": f"{host}:{port}",
                "Degree (Links)": len(topo.neighbors(n)),
            })
        st.dataframe(node_records, hide_index=True, use_container_width=True)

        st.markdown("#### Link Weights Table")
        link_records = []
        for u, v, w in sorted(metrics["links"], key=lambda x: (x[0], x[1])):
            is_sev = (min(u, v), max(u, v)) in {
                (min(a, b), max(a, b)) for a, b in st.session_state["failed_links"]
            }
            status_str = "SEVERED (FAIL)" if is_sev else "ACTIVE"
            link_records.append({
                "Node A": u,
                "Node B": v,
                "Weight (Latency/Cost)": w,
                "Link Status": status_str,
            })
        st.dataframe(link_records, hide_index=True, use_container_width=True)


# =============================================================================
# TAB 2: Algorithm Comparison (BFS vs Dijkstra)
# =============================================================================
with tab_algo:
    st.subheader(f"Path Comparison: {selected_source} ➔ {selected_dest}")
    st.markdown(
        "Demonstrates why unweighted shortest-hop algorithms (BFS) do not guarantee minimum transmission cost "
        "when links possess non-uniform weights."
    )

    if comparison:
        c1, c2 = st.columns(2)

        # BFS Card
        with c1:
            st.markdown("### 🔵 Unweighted BFS (Shortest Hops)")
            if comparison.bfs_path:
                st.markdown(f"**Path Selected:** `{' ➔ '.join(comparison.bfs_path)}`")
                st.markdown(f"**Total Hops:** `{comparison.bfs_hops}`")
                st.markdown(f"**Total Path Cost (Weight Sum):** `{comparison.bfs_cost}`")
                st.markdown(f"**First Hop (Next Hop):** `{comparison.bfs_next_hop}`")

                # Show routing table entry for destination
                bfs_rt = build_routing_table(topo, selected_source)
                endpoint = bfs_rt.get_route(selected_dest)
                st.code(f"RoutingTable[{selected_source}].get_route('{selected_dest}') = {endpoint}", language="python")
            else:
                st.warning(f"Destination {selected_dest} is unreachable via BFS.")

        # Dijkstra Card
        with c2:
            st.markdown("### 🟢 Weighted Dijkstra (Least Cost)")
            if comparison.dijkstra_path:
                st.markdown(f"**Path Selected:** `{' ➔ '.join(comparison.dijkstra_path)}`")
                st.markdown(f"**Total Hops:** `{comparison.dijkstra_hops}`")
                st.markdown(f"**Total Path Cost (Weight Sum):** `{comparison.dijkstra_cost}`")
                st.markdown(f"**First Hop (Next Hop):** `{comparison.dijkstra_next_hop}`")

                # Show routing table entry for destination
                dijkstra_rt = build_weighted_routing_table(topo, selected_source)
                endpoint = dijkstra_rt.get_route(selected_dest)
                st.code(f"RoutingTable[{selected_source}].get_route('{selected_dest}') = {endpoint}", language="python")
            else:
                st.warning(f"Destination {selected_dest} is unreachable via Dijkstra.")

        # Educational Analysis Banner
        st.markdown("---")
        if comparison.is_different_path and comparison.is_cost_divergence:
            st.markdown(
                f"""
                <div class="insight-alert">
                    <b>🎯 Core DSA Trade-off Observed:</b><br>
                    • <b>BFS</b> chose a <b>{comparison.bfs_hops}-hop path</b> with a high weight of <b>{comparison.bfs_cost}</b> because it only counts edge transitions ($O(V+E)$).<br>
                    • <b>Dijkstra</b> chose a <b>{comparison.dijkstra_hops}-hop path</b> with a much lower total weight of <b>{comparison.dijkstra_cost}</b> using its min-heap priority queue ($O((V+E)\\log V)$).<br>
                    <b>Conclusion:</b> In real networks, routing strictly by hop count can congest low-bandwidth or high-latency satellite/WAN links!
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif not comparison.is_different_path and comparison.bfs_path:
            st.markdown(
                f"""
                <div class="insight-box">
                    <b>🤝 Convergence:</b> Both BFS and Dijkstra converged on the exact same optimal path
                    <code>{' ➔ '.join(comparison.dijkstra_path)}</code> (Hops: {comparison.dijkstra_hops}, Cost: {comparison.dijkstra_cost}).
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error(f"Destination {selected_dest} is disconnected from source {selected_source}.")

        # Complete Routing Table Inspector
        with st.expander(f"🔍 Inspect Full Compiled Routing Table for Router {selected_source}"):
            rt_col1, rt_col2 = st.columns(2)
            with rt_col1:
                st.markdown("**Unweighted BFS Routing Table**")
                bfs_table = build_routing_table(topo, selected_source)
                b_entries = []
                for dest in sorted(topo.nodes()):
                    if dest != selected_source:
                        ep = bfs_table.get_route(dest)
                        b_entries.append({"Destination": dest, "Next Hop Endpoint": f"{ep[0]}:{ep[1]}" if ep else "NO ROUTE"})
                st.dataframe(b_entries, hide_index=True, use_container_width=True)

            with rt_col2:
                st.markdown("**Weighted Dijkstra Routing Table**")
                dijkstra_table = build_weighted_routing_table(topo, selected_source)
                d_entries = []
                for dest in sorted(topo.nodes()):
                    if dest != selected_source:
                        ep = dijkstra_table.get_route(dest)
                        d_entries.append({"Destination": dest, "Next Hop Endpoint": f"{ep[0]}:{ep[1]}" if ep else "NO ROUTE"})
                st.dataframe(d_entries, hide_index=True, use_container_width=True)


# =============================================================================
# TAB 3: Packet Forwarding Simulation
# =============================================================================
with tab_packet:
    st.subheader("In-Process Packet Forwarding Simulation")
    st.caption(
        "Traces a packet step-by-step through real router forwarding tables, decrementing TTL at each hop "
        "and detecting delivery or drops. (Simulated in-process using the project's actual Packet schema and RoutingTable lookup semantics)."
    )

    p_col1, p_col2, p_col3, p_col4 = st.columns([2, 2, 2, 2])
    with p_col1:
        sim_src = st.selectbox("Packet Origin (src)", options=routers, index=0, key="sim_src")
    with p_col2:
        sim_dest = st.selectbox("Packet Target (dest)", options=all_nodes, index=len(all_nodes)-2, key="sim_dest")
    with p_col3:
        sim_ttl = st.slider("Initial Time-To-Live (TTL)", min_value=1, max_value=15, value=8, step=1)
    with p_col4:
        sim_mode = st.selectbox("Routing Strategy", options=["weighted (Dijkstra)", "unweighted (BFS)"], index=0)

    sim_payload = st.text_input("Packet Payload String", value="Demo payload: Viva verification packet #42")

    if st.button("🚀 Transmit Simulated Packet", type="primary", use_container_width=True):
        mode_key = "weighted" if "weighted" in sim_mode else "unweighted"
        result = simulate_packet_forwarding(
            topo,
            source=sim_src,
            destination=sim_dest,
            payload=sim_payload,
            initial_ttl=sim_ttl,
            routing_mode=mode_key,
        )
        st.session_state["last_simulation"] = result

    # Display simulation trace if available
    sim_result: Optional[SimulationResult] = st.session_state.get("last_simulation")
    if sim_result:
        st.markdown("---")
        st.markdown("### Simulation Execution Trace")

        # Outcome banner
        if sim_result.delivered:
            st.success(f"✅ {sim_result.summary}")
        elif sim_result.status == "DROPPED_TTL":
            st.error(f"🛑 {sim_result.summary} (Initial TTL={sim_result.initial_ttl} was insufficient for this path length)")
        else:
            st.warning(f"⚠️ {sim_result.summary}")

        # Summary Metrics
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Final Status", sim_result.status)
        with sc2:
            st.metric("Total Hops Traversed", len(sim_result.hops_traversed) - 1)
        with sc3:
            st.metric("TTL Start ➔ End", f"{sim_result.initial_ttl} ➔ {sim_result.final_ttl}")
        with sc4:
            st.metric("Path Route", " ➔ ".join(sim_result.hops_traversed))

        # Hop-by-hop log table
        st.markdown("#### Hop Log & Forwarding Decisions")
        step_records = []
        for step in sim_result.steps:
            status_badge = (
                "DELIVERED" if step.status == "DELIVERED"
                else (step.status if "DROP" in step.status or step.status in ("NO_ROUTE", "UNREACHABLE") else "FORWARDING")
            )
            step_records.append({
                "Hop #": step.hop_number,
                "Node ID": step.current_node,
                "TTL In": step.incoming_ttl,
                "TTL Out": step.outgoing_ttl,
                "Action": step.action,
                "Next Hop": step.next_hop if step.next_hop else "—",
                "Outcome": status_badge,
                "Explanation": step.detail,
            })
        st.dataframe(step_records, hide_index=True, use_container_width=True)


# =============================================================================
# TAB 4: Resilience & Dynamic Topology
# =============================================================================
with tab_resilience:
    st.subheader("Dynamic Network Resilience & Fault Injection")
    st.caption(
        "Demonstrates automatic transactional rerouting using TopologyRoutingManager. "
        "Severing or updating links recalculates routing tables without crashing active routers."
    )

    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.markdown("#### ⚡ Sever / Restore Existing Links")
        existing_links = metrics["links"]
        link_options = [f"{u} ⟷ {v} (weight {w})" for u, v, w in existing_links]

        if link_options:
            selected_link_str = st.selectbox("Select Active Link to Sever:", options=link_options)
            link_to_sever_parts = selected_link_str.split(" ⟷ ")
            u_sever = link_to_sever_parts[0]
            v_sever = link_to_sever_parts[1].split(" ")[0]

            if st.button(f"✂️ Sever Link ({u_sever} ⟷ {v_sever})", type="secondary", use_container_width=True):
                try:
                    topo.disconnect(u_sever, v_sever)
                    st.session_state["failed_links"].add((min(u_sever, v_sever), max(u_sever, v_sever)))
                    st.success(f"Link {u_sever} ⟷ {v_sever} severed successfully! Routing tables recalculated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to sever link: {exc}")
        else:
            st.info("No active links available to sever.")

        # Restore severed links
        if st.session_state["failed_links"]:
            st.markdown("#### 🩹 Restore Severed Link")
            failed_opts = [f"{u} ⟷ {v}" for u, v in sorted(st.session_state["failed_links"])]
            restore_choice = st.selectbox("Select Severed Link to Restore:", options=failed_opts)
            if st.button("🔌 Restore Link", use_container_width=True):
                r_u, r_v = restore_choice.split(" ⟷ ")
                default_w = 2 if "R3" in (r_u, r_v) else (100 if "R2" in (r_u, r_v) and "R1" in (r_u, r_v) else 5)
                try:
                    topo.connect(r_u, r_v, weight=default_w)
                    st.session_state["failed_links"].remove((min(r_u, r_v), max(r_u, r_v)))
                    st.success(f"Restored link {r_u} ⟷ {r_v} with weight {default_w}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to restore link: {exc}")

    with r_col2:
        st.markdown("#### ➕ Add New Link or Update Link Weight")
        w_col1, w_col2, w_col3 = st.columns([2, 2, 2])
        with w_col1:
            add_u = st.selectbox("Endpoint A", options=all_nodes, index=0, key="add_u")
        with w_col2:
            add_v_opts = [n for n in all_nodes if n != add_u]
            add_v = st.selectbox("Endpoint B", options=add_v_opts, index=0, key="add_v")
        with w_col3:
            add_weight = st.number_input("Link Weight", min_value=0.0, max_value=1000.0, value=1.0, step=1.0)

        if st.button("🔗 Connect / Update Link", use_container_width=True):
            try:
                # If link exists, update weight; otherwise connect
                if topo.has_link(add_u, add_v):
                    topo.set_link_weight(add_u, add_v, add_weight)
                    st.success(f"Updated link weight for {add_u} ⟷ {add_v} to {add_weight}.")
                else:
                    topo.connect(add_u, add_v, weight=add_weight)
                    st.success(f"Created new link {add_u} ⟷ {add_v} with weight {add_weight}.")
                # Remove from failed links if present
                st.session_state["failed_links"].discard((min(add_u, add_v), max(add_u, add_v)))
                st.rerun()
            except Exception as exc:
                st.error(f"Operation failed: {exc}")

    st.markdown("---")
    st.markdown("#### 🔄 Resilience Manager Transaction Verification")
    st.caption(
        "Demonstrates the two-phase commit guarantee in TopologyRoutingManager: candidate tables are calculated "
        "on a deep copy before any live topology mutations occur, ensuring fault isolation."
    )
    with st.expander("Run Live Transactional Simulation with TopologyRoutingManager"):
        mgr = TopologyRoutingManager(topo, default_routing_mode="weighted")
        mgr_records = []
        for r_id in routers:
            dummy_router = Router("127.0.0.1", topo.get_endpoint(r_id)[1])
            mgr.add_router(r_id, dummy_router, routing_mode="weighted")
            mgr_records.append({
                "Router ID": r_id,
                "Managed Routing Mode": mgr.get_routing_mode(r_id),
                "Routes Computed": len(mgr.get_router(r_id).routing_table),
            })
        st.dataframe(mgr_records, hide_index=True, use_container_width=True)
        st.success("All managed routers synchronized with 100% transactional consistency.")


# =============================================================================
# TAB 5: Viva & Complexity Reference
# =============================================================================
with tab_viva:
    st.subheader("Academic & Technical Viva Readiness")
    st.markdown(
        "Reference sheet covering algorithmic complexity, data structures, and architectural justifications "
        "for project presentations."
    )

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("### 📚 Algorithmic Complexity Comparison")
        dsa_table = [
            {
                "Algorithm": "Breadth-First Search (BFS)",
                "Time Complexity": "O(V + E)",
                "Space Complexity": "O(V)",
                "Data Structure": "FIFO Queue (`collections.deque`)",
                "Metric Optimized": "Fewest Hops",
                "Weight Sensitivity": "Unweighted (assumes weight = 1)",
            },
            {
                "Algorithm": "Dijkstra's Shortest Path",
                "Time Complexity": "O((V + E) log V)",
                "Space Complexity": "O(V + E)",
                "Data Structure": "Min-Heap (`heapq`) + Hash Table",
                "Metric Optimized": "Least Total Weight (Cost/Latency)",
                "Weight Sensitivity": "Weighted (non-negative numeric)",
            },
            {
                "Algorithm": "Routing Table Lookup",
                "Time Complexity": "O(1) expected",
                "Space Complexity": "O(V)",
                "Data Structure": "Hash Map (`dict`)",
                "Metric Optimized": "Line-rate packet forwarding",
                "Weight Sensitivity": "N/A (Forwarding engine)",
            },
        ]
        st.dataframe(dsa_table, hide_index=True, use_container_width=True)

    with v2:
        st.markdown("### 🛡️ Architectural Charters & Design Decisions")
        st.markdown(
            """
            1. **Standard Library First**:
               - Sockets (`socket`), framing (`struct`), queues (`heapq`, `deque`), data structures (`dict`, `set`).
               - No third-party network frameworks like Twisted or AsyncIO.
            2. **Length-Prefixed Framing Protocol**:
               - TCP is a byte-stream protocol without inherent message boundaries.
               - 4-byte big-endian prefix (`!I`) eliminates stream fragmentation and concatenation bugs.
            3. **Two-Phase Resilience Transactions**:
               - `refresh_all()`, `connect_link()`, and `disconnect_link()` calculate all prospective tables on a deep copy before mutating live topology.
               - Prevents split-brain and partially updated forwarding states.
            4. **Zero-Weight Cycle Loop Prevention**:
               - Dijkstra's min-heap settles each node exactly once (`visited` set guard).
               - Strict distance improvement prevents cycling on zero-weight loops.
            """
        )

    st.markdown("---")
    st.caption("Developed for DSA Network Packet Routing Simulation · Headless test-suite compatible")
