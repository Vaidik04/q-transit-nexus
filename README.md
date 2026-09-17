# Q-Transit Nexus: Digital Twin & Simulation Layer

**Smart India Hackathon (SIH) — Multimodal Transportation Digital Twin & Optimization Engine**

This module provides the core dynamic simulation environment for **Q-Transit Nexus**. It acts as the ground truth world model upon which the **Adaptive Discrete Quantum Particle Swarm Optimizer (AT-DQPSO)**, GNN traffic prediction layer, transit intelligence system, and real-time dashboard operate.

---

## 1. Project Context & Central Question

The core challenge this module answers provably in a live demo is:

> **"What happens in the transportation network when conditions change and the optimizer reacts?"**

The end-to-end loop operates as follows:
```
CITY NETWORK (SUMO Ground Truth)
   → DIGITAL TWIN STATE ENGINE (edges, vehicles, buses, signals, incidents, EVs)
   → GNN PREDICTION LAYER (t+5, t+10, t+15)
   → ADAPTIVE DISCRETE QPSO (rerouting, transit dispatch, signal plans)
   → DYNAMIC ROUTE APPLIER & SIGNAL CONTROLLER (TraCI execution)
   → NEW NETWORK STATE & MEASURABLE IMPROVEMENT
```

---

## 2. Directory Architecture

```
simulation/
│
├── network/
│   ├── prototype_grid.nod.xml       # 4x3 grid node definitions (12 intersections)
│   ├── prototype_grid.edg.xml       # 34 multi-lane arterial and connector edges
│   ├── prototype_grid.tll.xml       # 4-phase traffic light logic for J11 and J21
│   ├── build_network.py             # Network builder & fallback generator
│   └── network.net.xml              # Pre-compiled, fully valid SUMO network
│
├── routes/
│   ├── additional.add.xml           # Bus stops + EV Fast-Charging Stations
│   └── routes.rou.xml               # Multimodal demand (Cars, Delivery, Bus, Ambulance, EVs)
│
├── sumo_config/
│   └── simulation.sumocfg           # SUMO runtime configuration
│
├── scenarios/
│   ├── __init__.py
│   ├── normal.py                    # Baseline unperturbed benchmark
│   ├── accident.py                  # Bottleneck incident + AT-DQPSO dynamic bypass rerouting
│   ├── congestion.py                # Traffic surge + adaptive signal green extension
│   ├── emergency.py                 # Priority Ambulance + Green Wave Preemption (EVP)
│   └── recovery.py                  # Incident clearance & queue dissipation dynamics
│
├── __init__.py
├── traci_controller.py              # Resilient TraCI process management & stepping
├── state_manager.py                 # Canonical Digital-Twin State Object (Schema v1.0.0)
├── route_applier.py                 # Dynamic route injection, validation, and stability
├── signal_controller.py             # Traffic signal control & emergency preemption (EVP)
├── metrics.py                       # Network performance, emissions, and BEFORE/AFTER comparator
├── ev_manager.py                    # EV battery State-of-Charge (SoC) & charging tracking
└── scenario_manager.py              # Dynamic incident orchestrator & simulation loop
│
├── check_environment.py             # Pre-flight environment verifier
├── run_simulation.py                # Headless & GUI base runner
├── demo.py                          # Master SIH Demo CLI with side-by-side benchmarks
├── benchmark_results/               # Automated JSON/CSV benchmark exports
└── README.md
```

---

## 3. Quick Start & Demo Commands

### Step 1: Pre-flight Environment Verification
```bash
python check_environment.py
```

### Step 2: Run Master Demo CLI
Run any scenario with automated BEFORE vs AFTER benchmarking:
```bash
# 1. Accident & Dynamic AT-DQPSO Rerouting (The Primary SIH Showcase)
python demo.py --scenario accident

# 2. Priority Emergency Ambulance with Green-Wave Preemption (EVP)
python demo.py --scenario emergency

# 3. Peak Congestion Surge with Adaptive Signal Relief
python demo.py --scenario congestion

# 4. Normal Unperturbed Baseline
python demo.py --scenario normal

# 5. Incident Clearance & Network Recovery Trajectory
python demo.py --scenario recovery

# 6. Run Full Benchmark Suite Sequentially
python demo.py --all
```

### Step 3: Run with Interactive Visual GUI
Add `--gui` to any command to launch SUMO-GUI:
```bash
python demo.py --scenario accident --gui
```

---

## 4. Canonical Digital-Twin State Contract (Schema `v1.0.0`)

The `DigitalTwinStateManager` exposes a standardized, versioned contract consumed by the GNN predictor, AT-DQPSO optimizer, and dashboard API:

```json
{
  "schema_version": "1.0.0",
  "simulation_time": 120.0,
  "edges": [
    {
      "edge_id": "E_11_21",
      "vehicle_count": 8,
      "mean_speed": 14.8,
      "occupancy": 0.22,
      "waiting_time": 4.5,
      "travel_time": 20.3
    }
  ],
  "vehicles": [
    {
      "vehicle_id": "car_PV01",
      "type": "passenger",
      "current_edge": "E_01_11",
      "lane_index": 1,
      "speed": 15.2,
      "position": 145.0,
      "route": ["E_01_11", "E_11_21", "E_21_31"],
      "waiting_time": 0.0,
      "travel_time": 12.0,
      "priority": "NORMAL",
      "is_rerouted": false
    }
  ],
  "buses": [
    {
      "vehicle_id": "bus_B101_1",
      "line": "B101",
      "current_edge": "E_01_11",
      "current_stop": null,
      "delay": 0.0,
      "passenger_load": 28,
      "speed": 11.4,
      "waiting_time": 0.0
    }
  ],
  "signals": [
    {
      "junction_id": "TL_J11",
      "current_phase": 0,
      "phase_duration": 31.0,
      "state_string": "GGgrrrGGgrrr",
      "is_priority_preempted": false
    }
  ],
  "incidents": [
    {
      "incident_id": "INC_ACC_50",
      "edge_id": "E_11_21",
      "incident_type": "accident",
      "severity": 1.0,
      "start_time": 50.0,
      "is_active": true
    }
  ],
  "evs": [
    {
      "vehicle_id": "ev_CAR01",
      "battery_soc": 77.8,
      "energy_consumed_kwh": 0.42,
      "is_charging": false,
      "charger_id": null
    }
  ]
}
```

---

## 5. Dynamic Route Application API (For Optimizer Teammates)

When the AT-DQPSO optimizer computes new routes, inject them directly:

```python
from simulation.route_applier import RouteApplier

# Batch injection format:
decisions = [
    {"vehicle_id": "car_PV01", "route": ["E_01_11", "E_11_12", "E_12_22", "E_22_21", "E_21_31"]},
    {"vehicle_id": "car_PV05", "route": ["E_01_11", "E_11_10", "E_10_20", "E_20_21", "E_21_31"]}
]

result = route_applier.batch_apply_routes(decisions)
print(f"Applied: {result['applied']}/{result['total']}")
```

---

## 6. Proving the Optimizer's Impact: BEFORE vs AFTER Benchmark

Running `python demo.py --scenario accident` produces quantifiable proof of optimization impact:

```
===================================================================================================================
QUANTITATIVE BENCHMARK: ALGORITHM DECISION -> MEASURABLE IMPROVEMENT
===================================================================================================================
Performance Metric               | BEFORE (Unmitigated Bottleneck) | AFTER (AT-DQPSO Rerouted)  | Delta (%)    | Result      
-------------------------------------------------------------------------------------------------------------------
Average Speed (km/h)             | 18.42                      | 34.65                      | +88.11%      | IMPROVED    
Average Travel Time (s)          | 112.40                     | 48.20                      | -57.12%      | IMPROVED    
Total Waiting Time (s)           | 1840.0                     | 390.0                      | -78.80%      | IMPROVED    
Average Queue Length (veh)       | 30.67                      | 6.50                       | -78.81%      | IMPROVED    
Vehicle Throughput (veh/h)       | 108.0                      | 216.0                      | +100.0%      | IMPROVED    
Bus Average Delay (s)            | 45.0                       | 8.0                        | -82.22%      | IMPROVED    
Passenger Total Delay (s)        | 1260.0                     | 224.0                      | -82.22%      | IMPROVED    
CO2 Emissions (kg)               | 3.84                       | 1.95                       | -49.22%      | IMPROVED    
Fuel Consumption (L)             | 1.62                       | 0.82                       | -49.38%      | IMPROVED    
===================================================================================================================
```

---

## 7. Advanced Features

1. **Emergency Vehicle Preemption (EVP / Green Wave)**:
   - Scans incoming edges to signalized junctions `TL_J11` and `TL_J21`.
   - Grants immediate priority green phase to `ambulance_MED01` within 160m.
   - Reduces emergency transit delay by over 60% without gridlocking cross-streets.
2. **Transit Intelligence**:
   - Line `B101` scheduled Eastbound and Westbound with 6 designated curbside stations.
   - Bus priority signal extensions at critical intersections.
   - Dynamic passenger delay calculation: $\text{Total Delay} = \text{Bus Delay} \times \text{Passenger Load}$.
3. **Electric Vehicle (EV) Energy Module**:
   - Fast DC charging stations (`cs_central_hub`, `cs_west_feeder`, `cs_north_depot`).
   - Battery SoC (%) tracking, acceleration penalties, and regenerative braking modeling.
