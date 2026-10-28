# DSA Network Packet Routing & Topology Simulation

An educational simulation of **Network Packet Routing** built using Python 3 and fundamental Data Structures & Algorithms (DSA). The project demonstrates how packets navigate network topologies using real TCP/IP sockets, custom 4-byte length-prefixed framing, routing tables, and graph pathfinding algorithms (Breadth-First Search and Dijkstra's algorithm).

---

## Quick Start & Setup

### 1. Prerequisites & Environment Setup

Python 3.9+ is recommended. Core algorithms and TCP networking modules rely **strictly on the Python Standard Library** (`socket`, `struct`, `json`, `heapq`, `collections`, `selectors`, `unittest`).

Clone the repository and navigate into the project directory:

**Bash (Linux / macOS / Git Bash):**
```bash
git clone https://github.com/msylvia212418-coder/dsa-project.git
cd dsa-project
```

**PowerShell (Windows):**
```powershell
git clone https://github.com/msylvia212418-coder/dsa-project.git
cd dsa-project
```

### 2. Dependency Installation

The interactive presentation dashboard requires Streamlit:

```bash
pip install -r requirements.txt
```

### 3. Launch the Interactive Dashboard

Run the Streamlit application from the project root:

```bash
python -m streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

### 4. Run Automated Test Suite

Execute the complete automated test suite headlessly:

```bash
python -m unittest discover -s NetworkPacketRouting/tests -v
```

---

## Dashboard Feature Guide

The dashboard is organized into four dedicated views:

1. **Network Topology**:
   - Visualizes routers (blue boxes) and receiver leaf nodes (green capsules).
   - Real-time link weights and latency costs displayed directly on edges.
   - Severed links rendered distinctly as dashed red lines.
   - Interactive path highlighter dynamically tracing the active route.
   - Tabular view of registered socket endpoints `(host, port)` and node degrees.

2. **Route Comparison**:
   - Side-by-side evaluation of **Unweighted BFS** (shortest hop) versus **Weighted Dijkstra** (least cost).
   - Analysis cards explaining why fewest-hop routes can route across high-latency or expensive links.
   - Clear distinction between cases where algorithms diverge and where they converge on the same path.
   - Collapsible forwarding table inspector displaying compiled next-hop mappings for the origin router.

3. **Packet Simulator**:
   - Hop-by-hop packet forwarding trace with per-hop TTL decrement tracking.
   - Displays canonical packet outcomes: `DELIVERED`, `DROPPED_TTL`, and `NO_ROUTE`.
   - Distinguishes simulation-only loop guard outcomes: `DROPPED_LOOP` (cycle detected) and `DROPPED_HOPS_EXCEEDED` (hop limit exceeded).
   - Visual summary displaying destination node, final status, traversal path, and initial vs. final TTL.
   - *Note*: Operates in-process using the project's canonical `Packet` dataclass and compiled `RoutingTable` lookups for step-by-step visual inspection.

4. **Resilience & Link Management**:
   - Fault injection: sever existing links to simulate network outages.
   - Link restoration: restore severed links with their original saved weights.
   - Parameter adjustment: update link weights or connect new nodes dynamically.
   - Real-time route impact view showing updated paths and costs immediately.

---

## Interactive Demonstration Workflow

Follow this step-by-step sequence to explore core routing behaviors:

### Step 1: Compare BFS vs Dijkstra Trade-offs
1. Open the dashboard (defaults to the Diamond demo topology).
2. Navigate to **Route Comparison**.
3. Select Source: `R1`, Destination: `Dest-Alpha`.
4. **Observe**:
   - **BFS** selects `R1 ➔ R2 ➔ Dest-Alpha`: **2 hops**, total weight = **101**.
   - **Dijkstra** selects `R1 ➔ R3 ➔ R2 ➔ Dest-Alpha`: **3 hops**, total weight = **5**.
   - Dijkstra selects a path with more hops because the cumulative link cost is significantly lower ($5$ vs $101$).

### Step 2: Simulate Packet Forwarding & TTL Mechanics
1. Navigate to **Packet Simulator**.
2. Set Source: `R1`, Destination: `Dest-Alpha`, Routing Strategy: `Weighted (Dijkstra)`.
3. Set Initial TTL = `8` and click **Transmit Packet**:
   - Packet traverses `R1 ➔ R3 ➔ R2 ➔ Dest-Alpha` and arrives with `final_ttl=5` (`DELIVERED`).
4. Set Initial TTL = `2` and click **Transmit Packet**:
   - Packet decrements TTL at `R1` ($2 \rightarrow 1$), forwards to `R3`, and drops at `R3` with `DROPPED_TTL` ($1 \le 1$).

### Step 3: Test Dynamic Rerouting on Link Severance
1. Navigate to **Resilience & Link Management**.
2. Under **Sever Link**, select `R1 ⟷ R3 (weight 2)` and click **Sever Link**.
3. Return to **Route Comparison** or **Network Topology**:
   - The primary low-cost path via `R3` is severed.
   - Dijkstra dynamically reroutes across the alternate multi-hop path via `R4`: `R1 ➔ R4 ➔ R2 ➔ Dest-Alpha` with total cost **11** (still lower than the $101$ direct link).
4. Return to **Resilience & Link Management**, select `R1 ⟷ R3` under **Restore Link**, and restore it to its original weight.

---

## Architecture & System Design

### In-Process Simulation vs Real TCP Socket Networking

The project deliberately separates visual simulation from real network transmission:

| Dimension | In-Process Dashboard Simulator | Headless TCP Socket Integration Suite |
| :--- | :--- | :--- |
| **Execution Model** | Synchronous, in-process traversal | Multi-threaded concurrent TCP servers (`socket.socket`) |
| **Forwarding Engine** | In-memory `RoutingTable.get_route()` lookups | Real network socket I/O over loopback interface (`127.0.0.1`) |
| **Packet Protocol** | In-memory `Packet` dataclass instances | Binary 4-byte big-endian framed JSON byte streams (`struct.pack("!I", len)`) |
| **Port Usage** | No physical network ports opened | Binds listening OS ports with `SO_REUSEADDR` |
| **Primary Purpose** | Visual step-by-step tracing and algorithmic comparison | Concurrency testing, socket error handling, and framing integrity |

### Resilience & Transactional Routing Updates

The `TopologyRoutingManager` orchestrates dynamic routing table updates across active routers:
- **Two-Phase Route Preparation**: Before mutating active routers or live links, candidate forwarding tables are computed against a copy of the topology. If route computation fails for any node, live topology and router states remain unchanged.
- **Controlled Application**: Prepared tables are installed across managed routers. If an installation step fails, a `ResilienceError` is raised documenting any partial update state.
- **Continuous Socket Operation**: Bound ports, listening sockets, and client worker threads stay running across routing table updates.

### Canonical Packet & Router Statuses

Defined in `NetworkPacketRouting.packet.VALID_STATUSES` and `NetworkPacketRouting.router`:
- `DELIVERED`: Packet successfully reached its target destination.
- `DROPPED_TTL`: Packet dropped because TTL was exhausted ($\le 1$) prior to forwarding.
- `NO_ROUTE`: Router had no matching forwarding entry for the target destination.
- `UNREACHABLE`: Downstream connection or forwarding failed due to network I/O error.
- `INVALID_PACKET`: Packet payload failed schema or format validation.

*Simulation-Only Outcomes* (`dashboard_controller.py`):
- `DROPPED_LOOP`: Forwarding cycle detected at an already-visited node.
- `DROPPED_HOPS_EXCEEDED`: Traversal exceeded maximum hop guard limit.

---

## Project Structure

```text
dsa-project/
├── app.py                                # Streamlit presentation dashboard
├── requirements.txt                      # Dashboard dependencies
├── AGENTS.md                             # Architectural charter and standards
└── NetworkPacketRouting/
    ├── graph.py                          # Adjacency-list graph, BFS, and Dijkstra
    ├── topology.py                       # Network topology and endpoint registry
    ├── route_planner.py                  # Forwarding table compilers (BFS & Dijkstra)
    ├── routing_table.py                  # In-memory routing table mapping dest -> endpoint
    ├── router.py                         # Single-router synchronous TCP forwarding engine
    ├── topology_router.py                # Factory creating routers from topology nodes
    ├── resilience.py                     # Multi-router resilience and route update manager
    ├── packet.py                         # Packet schemas, validation, and JSON serialization
    ├── protocol.py                       # 4-byte big-endian length-prefixed socket framing
    ├── dashboard_controller.py           # Dashboard controller and simulation engine
    └── tests/                            # Headless test suite (305 passing tests)
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

## Algorithmic Complexity

| Component / Operation | Algorithm | Time Complexity | Space Complexity | Data Structures Used |
| :--- | :--- | :--- | :--- | :--- |
| **Shortest-Hop Pathfinding** | Breadth-First Search (BFS) | $O(V + E)$ | $O(V)$ | FIFO Queue (`collections.deque`), Visited Set |
| **Least-Cost Pathfinding** | Dijkstra's Algorithm | $O((V + E) \log V)$ | $O(V + E)$ | Min-Heap Priority Queue (`heapq`), Distance Map |
| **Forwarding Table Compilation** | Full Topology Scan | $O(V \cdot (V + E) \log V)$ | $O(V^2)$ | Adjacency List, Intermediate Path Dicts |
| **Route Lookup** | Hash Table Lookup | $O(1)$ average | $O(V)$ | Python Hash Map (`dict`) |
| **Frame Demarcation** | Length-Prefixed Header | $O(1)$ | $O(1)$ | 4-byte Big-Endian Prefix (`struct.pack("!I")`) |
