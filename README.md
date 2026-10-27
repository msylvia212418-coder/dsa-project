# DSA Network Packet Routing & Topology Simulation

A production-grade, educational simulation of **Network Packet Routing** built using Python 3 and fundamental Data Structures & Algorithms (DSA). The project demonstrates how packets navigate complex network topologies using real TCP/IP sockets, custom 4-byte length-prefixed framing, routing tables, and graph algorithms (Breadth-First Search and Dijkstra's algorithm).

---

## 🚀 Quick Start & Dashboard Launch

### 1. Environment Setup

Python 3.9+ is recommended. Core algorithms and TCP networking modules rely **strictly on the Python Standard Library** (`socket`, `struct`, `json`, `heapq`, `collections`, `selectors`, `unittest`).

Clone the repository and navigate into the project directory:
```bash
git clone https://github.com/msylvia212418-coder/dsa-project.git
cd dsa-project
```

### 2. Dependency Installation

The interactive presentation dashboard uses Streamlit:
```bash
pip install -r requirements.txt
```

### 3. Launch the Interactive Dashboard

Launch the visual dashboard locally with a single command:
```bash
python -m streamlit run app.py
```
This automatically opens the dashboard in your default browser at `http://localhost:8501`.

---

## 🎯 Dashboard Feature Guide

The dashboard is designed for viva presentations and interactive demonstrations:

1. **🗺️ Interactive Network Topology Map**:
   - Visualizes routers (blue boxes) and receiver leaf nodes (green capsules).
   - Real-time link weights and latency costs displayed directly on edges.
   - Live endpoint table displaying `(host, port)` bindings and node degree.

2. **⚡ Algorithm Comparison (BFS vs Dijkstra)**:
   - Side-by-side evaluation of **Unweighted BFS** (shortest hop) versus **Weighted Dijkstra** (least cost).
   - Educational insight cards explaining why fewest-hop routes can congest low-bandwidth / high-latency links.
   - Interactive path highlighter tracing the selected path directly on the topology graph.

3. **📦 Packet Forwarding Simulation**:
   - Trace packet traversal hop-by-hop with real-time TTL decrementing.
   - Covers all network outcomes: `DELIVERED`, `DROPPED_TTL`, and `DROPPED_NO_ROUTE`.
   - *Note*: Operates in-process using the project's actual `Packet` schema and compiled `RoutingTable` lookups for sub-millisecond visual step-by-step tracing.

4. **🛡️ Dynamic Topology Resilience & Fault Injection**:
   - Dynamically sever links to simulate network outages.
   - Observe instantaneous, transactional rerouting powered by `TopologyRoutingManager`.
   - Add new links or adjust link weights with live recalculations.

5. **📊 Viva & Complexity Reference**:
   - Reference table summarizing time and space complexities ($O(V+E)$ BFS vs $O((V+E)\log V)$ Dijkstra vs $O(1)$ routing table lookups).
   - Architectural decisions defensible in technical and academic reviews.

---

## 🔬 Presentation Demo Workflow

Follow this step-by-step workflow during a demonstration:

### Step 1: Demonstrate BFS vs Dijkstra Trade-off
1. Open the dashboard (defaults to the Diamond demo topology).
2. Go to **⚡ Algorithm Comparison**.
3. Select Source: `R1`, Destination: `Dest-Alpha`.
4. **Observe**:
   - **BFS** selects `R1 ➔ R2 ➔ Dest-Alpha`: **2 hops**, total cost = **101**.
   - **Dijkstra** selects `R1 ➔ R3 ➔ R2 ➔ Dest-Alpha`: **3 hops**, total cost = **5**.
   - Point out that Dijkstra chose a path with more hops because it is **20x cheaper** ($5$ vs $101$).

### Step 2: Simulate Packet Forwarding & TTL Expiry
1. Go to **📦 Packet Forwarding Simulation**.
2. Select Source: `R1`, Destination: `Dest-Alpha`, Routing Strategy: `weighted (Dijkstra)`.
3. Set Initial TTL = `8`, click **Transmit Simulated Packet**.
   - Watch the packet traverse `R1 ➔ R3 ➔ R2 ➔ Dest-Alpha` and arrive with `TTL=5` (`DELIVERED`).
4. Now change Initial TTL to `2` and transmit again.
   - Watch the packet get dropped at `R3` with `DROPPED_TTL` because TTL expired before reaching destination.

### Step 3: Simulate a Link Failure & Automatic Rerouting
1. Go to **🛡️ Resilience & Dynamic Topology**.
2. Under **Sever / Restore Existing Links**, select link `R1 ⟷ R3 (weight 2)` and click **✂️ Sever Link**.
3. Switch back to **⚡ Algorithm Comparison** or **🗺️ Network Topology**.
4. **Observe**:
   - The primary cheap route is severed.
   - Dijkstra automatically reroutes packets along the alternate path via `R4`: `R1 ➔ R4 ➔ R2 ➔ Dest-Alpha` with cost **11** (still much cheaper than the $101$ direct link).

---

## 🧪 Testing & Validation

The project includes an automated test suite with **over 300 tests**:

```bash
# Run all unit and integration tests headlessly
python -m unittest discover -s NetworkPacketRouting/tests -v
```

### Distinction: Simulation vs Real TCP Integration Tests
- **In-Dashboard Packet Simulation**:
  Runs in-process using the compiled `RoutingTable` and `Packet` objects to visualize hop-by-hop decisions without opening physical network ports.
- **Headless TCP Integration Tests**:
  Files such as `test_topology_router.py` and `test_resilience.py` start genuine, concurrent TCP server threads (`socket.socket`, `SO_REUSEADDR`, 4-byte length-prefixed protocol frames) and transmit real binary byte streams over loopback (`127.0.0.1`), verifying fault tolerance, socket closure, and real TCP packet delivery under simulated link failure and zero-weight cycles.

---

## 🏛️ Project Architecture

```text
dsa-project/
├── app.py                                # Streamlit visual presentation dashboard
├── requirements.txt                      # Dashboard dependencies
├── AGENTS.md                             # Architectural contract and charter
└── NetworkPacketRouting/
    ├── graph.py                          # Adjacency-list graph + BFS + Dijkstra
    ├── topology.py                       # Network topology and endpoint registry
    ├── route_planner.py                  # Forwarding table compilers (BFS & Dijkstra)
    ├── routing_table.py                  # In-memory routing table mapping dest -> endpoint
    ├── router.py                         # Single-router TCP forwarding engine
    ├── topology_router.py                # Factory creating routers from topology nodes
    ├── resilience.py                     # Multi-router transactional resilience manager
    ├── packet.py                         # Packet schema, validation, JSON serialization
    ├── protocol.py                       # 4-byte big-endian length-prefixed socket framing
    ├── dashboard_controller.py           # Dashboard controller & simulation services
    └── tests/                            # Comprehensive headless test suite (302 tests)
        ├── test_graph.py
        ├── test_topology.py
        ├── test_route_planner.py
        ├── test_routing_table.py
        ├── test_router.py
        ├── test_topology_router.py
        ├── test_resilience.py
        ├── test_packet.py
        ├── test_protocol.py
        └── test_dashboard_controller.py
```

---

## 📜 Academic Integrity & Engineering Standards
- **Standard Library First**: No external networking dependencies for core logic.
- **Framing Discipline**: 4-byte length prefix prevents TCP streaming fragmentation bugs.
- **Robustness**: Graceful error handling for TTL drops, disconnected components, and socket drops.
- **Big-O Optimal**: $O((V+E)\log V)$ shortest paths, $O(1)$ table lookups.
