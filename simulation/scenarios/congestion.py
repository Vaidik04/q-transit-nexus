"""
Scenario: simulation.scenarios.congestion
Description:
    Simulates a peak traffic congestion surge across the central urban network.
    Compares fixed-time traffic light cycles against adaptive signal green extensions
    that dynamically clear arterial queues and reduce commuter delays.

Usage:
    python -m simulation.scenarios.congestion
    python -m simulation.scenarios.congestion --steps 300
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.traci_controller import TraCIController
from simulation.state_manager import DigitalTwinStateManager
from simulation.route_applier import RouteApplier
from simulation.signal_controller import SignalController
from simulation.metrics import MetricsCollector, compare_metrics
from simulation.ev_manager import EVManager
from simulation.scenario_manager import ScenarioManager

def run_congestion_pass(steps: int, adaptive_signals: bool, gui: bool = False):
    config_path = os.path.join(PROJECT_ROOT, "simulation", "sumo_config", "simulation.sumocfg")
    controller = TraCIController(config_path, gui=gui)
    ev_mgr = EVManager(controller)
    state_mgr = DigitalTwinStateManager(controller, ev_mgr=ev_mgr)
    route_applier = RouteApplier(controller)
    signal_ctrl = SignalController(controller)
    metrics_col = MetricsCollector(controller)
    scenario_mgr = ScenarioManager(controller, state_mgr, route_applier, signal_ctrl, metrics_col, ev_mgr)

    mode_label = "WITH ADAPTIVE SIGNAL MANAGEMENT" if adaptive_signals else "WITH FIXED-TIME SIGNALS (CONGESTION PEAK)"
    print(f"\n[RUN] Starting pass: {mode_label}...")

    controller.start()
    traci = controller.traci

    def step_hook(sim_time: float):
        # Adaptive signal logic: if queue on E_01_11 or E_11_21 exceeds threshold, extend green phase
        if adaptive_signals and sim_time % 10 == 0:
            try:
                occ_1 = traci.edge.getLastStepOccupancy("E_01_11")
                occ_2 = traci.edge.getLastStepOccupancy("E_11_21")
                if occ_1 > 0.40:
                    # Extend East-West arterial green at TL_J11
                    cur_p = traci.trafficlight.getPhase("TL_J11")
                    if cur_p == 0:  # EW Green
                        traci.trafficlight.setPhaseDuration("TL_J11", 45.0)
                if occ_2 > 0.40:
                    cur_p = traci.trafficlight.getPhase("TL_J21")
                    if cur_p == 0:
                        traci.trafficlight.setPhaseDuration("TL_J21", 45.0)
            except Exception:
                pass

    try:
        summary = scenario_mgr.run_simulation_loop(
            total_steps=steps,
            step_hook=step_hook,
            enable_evp=True,
            enable_ev=True
        )
        return summary
    finally:
        controller.close()

def run_congestion_benchmark(steps: int = 300, gui: bool = False):
    print("=" * 75)
    print("Q-TRANSIT NEXUS: Congestion Surge & Adaptive Signal Benchmark")
    print("=" * 75)
    print("Hypothesis: Under peak traffic conditions, dynamically adjusting signal")
    print("phase durations based on edge occupancy flushes arterial queues and")
    print("stabilizes overall network throughput.")
    print("-" * 75)

    metrics_before = run_congestion_pass(steps=steps, adaptive_signals=False, gui=gui)
    metrics_after = run_congestion_pass(steps=steps, adaptive_signals=True, gui=gui)

    comparison = compare_metrics(
        before=metrics_before,
        after=metrics_after,
        label_before="BEFORE (Fixed-Time Signals)",
        label_after="AFTER (Adaptive Signal Relief)"
    )

    print("\n" + "=" * 115)
    print("CONGESTION RELIEF BENCHMARK: FIXED VS ADAPTIVE CONTROL")
    print("=" * 115)
    print(comparison["formatted_table"])
    print("=" * 115)

    results_dir = os.path.join(PROJECT_ROOT, "benchmark_results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "congestion_comparison.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"[SUCCESS] Benchmark report written to: {out_file}\n")

    return comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Congestion Surge Scenario")
    parser.add_argument("--steps", type=int, default=300, help="Simulation duration (default: 300s)")
    parser.add_argument("--gui", action="store_true", help="Launch in SUMO GUI")
    args = parser.parse_args()

    run_congestion_benchmark(steps=args.steps, gui=args.gui)
