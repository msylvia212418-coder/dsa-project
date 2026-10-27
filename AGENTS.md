# AGENTS.md - Engineering Contract & Architecture Charter

## 1. Project Charter & Scope
This project is an educational, production-grade simulation of **Network Packet Routing** built using Python 3 and fundamental Data Structures & Algorithms (DSA). It demonstrates how packets navigate a simulated network topology using real TCP/IP sockets, custom packet framing, routing tables, and graph algorithms.

### Milestones Roadmap
1. **Milestone 0: Baseline & Repository Foundation** (Current)
   - Version-controlled baseline, `.gitignore`, architecture contract, and clean development workflow.
2. **Milestone 1: Single-Router TCP Forwarding**
   - Robust packet protocol with length-prefixed framing.
   - Header validation: version, source, destination, packet ID, TTL, payload.
   - Deterministic route lookup via static routing table.
   - Single-router TCP forwarding engine with graceful error handling (invalid packets, missing routes, TTL expiry do not crash the router).
   - Automated unit tests and end-to-end socket integration tests.
3. **Milestone 2: Custom Graph Topology & Pathfinding (BFS / DFS)**
   - Custom adjacency-list graph representation implemented from scratch (no external graph libraries).
   - Breadth-First Search (BFS) for shortest-hop path calculation.
   - Depth-First Search (DFS) for path discovery and cycle detection.
4. **Milestone 3: Dynamic Multi-Router Mesh & Metrics (Dijkstra / Weighted Graphs)**
   - Multi-hop routing across interconnected router nodes.
   - Dijkstra's algorithm for least-latency / least-cost paths.
   - Metrics: packet loss, hop count, transmission delay, queue depth.

---

## 2. Engineering Standards & Rules of Engagement

### Core Constraints
- **Standard Library First**: Rely strictly on Python standard library modules (`socket`, `struct`, `json`, `selectors`, `time`, `unittest`, `dataclasses`, `typing`). No third-party network frameworks unless explicitly approved.
- **Robustness & Fault Tolerance**:
  - Sockets must use proper resource cleanup (`try...finally` or context managers).
  - Malformed packets, socket resets, or missing routing table entries must log warnings/errors and drop the packet gracefully without crashing the server process.
  - Sockets must set `SO_REUSEADDR` to allow rapid restart during tests and development.
- **Protocol Discipline**:
  - Sockets communicate byte streams, not discrete messages. All communications must use framing (e.g., 4-byte big-endian length prefix followed by encoded payload) to prevent fragmentation/concatenation bugs.
  - Every packet carries standard metadata: `version`, `src`, `dest`, `packet_id`, `ttl`, and `payload`.
- **Testing Standard**:
  - Every milestone must deliver comprehensive unit tests and automated end-to-end integration tests.
  - Test suites must be executable headlessly (`python -m unittest discover`).
  - No claims of test success without verified execution.

### Coding Style & Viva Readiness
- Code must be clear, idiomatic, well-documented, and defensible in academic/technical reviews (viva).
- Data structures must be cleanly separated from network I/O logic to allow algorithmic analysis (Big-O time and space complexity).

---

## 3. Git & Operational Discipline
- **Approval Gate**: All file additions, staging, and commits must be reviewed and approved by the project owner.
- **Atomic Commits**: Small, single-purpose commits with conventional commit messages (e.g., `feat:`, `fix:`, `test:`, `docs:`).
- **Non-Destructive Operations**: Never perform hard resets, rebase remote branches, delete files without approval, or push without confirmation.
