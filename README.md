# Q-TRANSIT NEXUS
> **Predictive Multimodal Digital Twin for Adaptive Quantum-Inspired Urban Transportation Optimization**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-cyan.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6.0-purple.svg)](https://vitejs.dev/)

---

## Overview

**Q-Transit Nexus** is an urban-scale, closed-loop transportation digital twin powered by **Adaptive Traffic-Aware Discrete Quantum Particle Swarm Optimization (AT-DQPSO)**. 

Unlike traditional routing engines that greedily optimize for current network conditions, Q-Transit Nexus forecasts spatial-temporal traffic conditions **5, 10, and 15 minutes ahead** and coordinates heterogeneous transportation agents (**logistics fleets, public transit buses, electric vehicles, emergency responders, and private traffic**) alongside physical infrastructure (**traffic signals, EV charging hubs, and dedicated priority corridors**).

```
                     CITY DATA (Real-time & Historical)
                                     │
                                     ↓
                          ML / SPATIO-TEMPORAL GNN
                         (T+5, T+10, T+15 Forecasts)
                                     │
                                     ↓
                           DIGITAL TWIN PLATFORM
                      (NetworkX + TraCI / SUMO Interface)
                                     │
                                     ↓
                          ADAPTIVE DISCRETE QPSO
                     min F = w₁T + w₂D + w₃C + w₄P + w₅E + w₆R + w₇V
                                     │
                 ┌───────────────────┼───────────────────┐
                 ↓                   ↓                   ↓
           Fleet Routing      Transit Intelligence   Signal & Emergency
           (Logistics/EVs)     (Buses/Bunching/TSP)     (Green Waves)
                 └───────────────────┬───────────────────┘
                                     ↓
                         CLOSED-LOOP ACTUATION & DT
                                     ↺
```

---

## Key Modules & Contributions

1. **AT-DQPSO Optimization Core** ([`backend/modules/at_dqpso.py`](file:///d:/Q-TransitNexus/backend/modules/at_dqpso.py))
   - Quantum delta potential well formulation with adaptive contraction-expansion coefficient ($\alpha$).
   - Multi-objective fitness function balancing Travel Time ($T$), Distance ($D$), Congestion ($C$), Passenger Delay ($P$), Emissions ($E$), Rerouting Instability ($R$), and Constraint Violations ($V$).
   - Dynamic operational modes: **Balanced City**, **Emergency Priority**, **Green Eco-Mode**, and **Public Transit Priority**.
2. **Spatio-Temporal ML Traffic Predictor** ([`backend/modules/traffic_prediction.py`](file:///d:/Q-TransitNexus/backend/modules/traffic_prediction.py))
   - Spatial-lag graph features predicting edge conditions at $T+5, T+10, T+15$ horizons.
   - Uncertainty estimation calculating 95th percentile travel time ($P_{95}$) for reliability-aware routing.
3. **Microscopic Digital Twin Engine** ([`backend/modules/simulation_engine.py`](file:///d:/Q-TransitNexus/backend/modules/simulation_engine.py))
   - TraCI-compatible discrete-event microscopic simulation tracking 31 heterogeneous vehicles, kinematic speed updates, and traffic signal cycles.
4. **Transit Intelligence & TSP** ([`backend/modules/transit_intelligence.py`](file:///d:/Q-TransitNexus/backend/modules/transit_intelligence.py))
   - Public transport passenger delay optimization ($P = \sum_i \text{passengers}_i \times \text{delay}_i$).
   - Bus bunching detection and headway deviation stabilization.
   - Transit Signal Priority (TSP) trading off passenger-minutes saved against cross-street delay.
5. **EV Co-Optimization** ([`backend/modules/ev_intelligence.py`](file:///d:/Q-TransitNexus/backend/modules/ev_intelligence.py))
   - Joint $(Route + ChargerSelection)$ co-optimization considering travel time and predicted charging station queues ($SOC_{arrival} \ge SOC_{min}$).
6. **Emergency Corridors** ([`backend/modules/emergency_priority.py`](file:///d:/Q-TransitNexus/backend/modules/emergency_priority.py))
   - Dedicated priority corridors and automated green waves for emergency responders.
7. **Explainable AI Engine** ([`backend/modules/explainability.py`](file:///d:/Q-TransitNexus/backend/modules/explainability.py))
   - Transparent decision rationales comparing previous path vs new path with exact empirical metrics.
8. **Transportation Trust Layer** ([`backend/modules/trust_layer.py`](file:///d:/Q-TransitNexus/backend/modules/trust_layer.py))
   - Cryptographic SHA-256 chained audit ledger for all optimization events and incident interventions.

---

## Comparative Benchmarks & Empirical Findings

Evaluated on the 24-node, 38-corridor urban digital twin with active incident bottleneck:

| Algorithm | Category | Travel Time | Distance | CO₂ Emissions | Runtime | Fitness Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dijkstra** | Greedy Shortest Path | 108.6 s | 1,650 m | 0.317 kg | 1.2 ms | 273.61 |
| **A\*** | Heuristic Search | 108.6 s | 1,650 m | 0.317 kg | 1.8 ms | 273.61 |
| **GA** | Genetic Algorithm | 108.6 s | 1,650 m | 0.317 kg | 41.0 ms | 438.61 |
| **ACO** | Ant Colony Optimization | 108.6 s | 1,650 m | 0.317 kg | 36.2 ms | 438.61 |
| **Standard QPSO**| Static Quantum Swarm | 108.6 s | 1,650 m | 0.317 kg | 153.3 ms | 438.61 |
| **AT-DQPSO (Ours)**| Adaptive Traffic-Aware Discrete QPSO | **91.4 s** | 1,250 m | **0.241 kg** | 24.5 ms | **7.98** |

*AT-DQPSO achieves a **15.8% reduction in travel time** and **24% lower CO₂ emissions** by dynamically routing around predicted bottlenecks.*

---

## Quickstart Guide

### 1. Requirements
- Python 3.9+
- Node.js 18+

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
cd dashboard
npm install
cd ..
```

### 3. Launch Backend API
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger documentation: `http://127.0.0.1:8000/docs`

### 4. Launch Command Center Dashboard
```powershell
cd dashboard
npm run dev
```
Dashboard: `http://127.0.0.1:5173/`

### 5. Run Reproducible Research Experiments
```powershell
# Run baseline comparison across all 6 algorithms
python experiments/benchmark.py

# Run module ablation study
python experiments/ablation.py

# Run custom parameter sweep
python run_experiment.py --scenario accident_corridor --vehicles 30 --algorithm AT-DQPSO --seed 100
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Core system status, version, and active vehicle count |
| `GET` | `/simulation/state` | Full digital twin state (nodes, edges, vehicles, incidents, signals) |
| `GET` | `/vehicles` | Active fleet state, positions, routes, and passenger loads |
| `GET` | `/traffic/current` | Live edge speeds, flows, and congestion indices |
| `POST`| `/prediction` | Spatio-temporal traffic forecasts ($T+5, T+10, T+15$) |
| `POST`| `/optimize` | Triggers AT-DQPSO fleet re-optimization |
| `POST`| `/incident` | Simulates accident on Road E14 / triggers automated closed-loop cascade |
| `POST`| `/simulation/start` | Starts continuous simulation animation loop |
| `POST`| `/simulation/stop` | Pauses continuous simulation loop |
| `GET` | `/metrics` | Global network KPIs (average speed, passenger delay, CO₂ savings) |
| `GET` | `/benchmark` | Comparative algorithm benchmarking telemetry |
| `GET` | `/ablation` | Module isolation performance data |
