"""
Master Demo Runner for Q-Transit Nexus Digital Twin & Simulation Layer.
Smart India Hackathon (SIH) Showcase CLI.

Usage:
    python demo.py --scenario normal       # Runs baseline traffic benchmark
    python demo.py --scenario accident     # Runs accident & AT-DQPSO dynamic reroute benchmark
    python demo.py --scenario congestion   # Runs congestion surge & adaptive signal benchmark
    python demo.py --scenario emergency    # Runs Code-3 Ambulance & EVP Green Wave benchmark
    python demo.py --scenario recovery     # Runs incident clearance & dissipation trajectory
    python demo.py --all                   # Runs complete benchmark suite sequentially
    python demo.py --scenario accident --gui  # Launches interactive SUMO GUI visualizer
"""

import os
import sys
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.scenarios.normal import run_normal_scenario
from simulation.scenarios.accident import run_accident_benchmark
from simulation.scenarios.congestion import run_congestion_benchmark
from simulation.scenarios.emergency import run_emergency_benchmark
from simulation.scenarios.recovery import run_recovery_scenario

def banner():
    print("""
=============================================================================
           ____       _______                    _ _     _   _                      
          / __ \     |__   __|                  (_) |   | \ | |                     
         | |  | |______| | |_ __ __ _ _ __  ___ _| |_   |  \| | _____  ___   _ ___  
         | |  | |______| | | '__/ _` | '_ \/ __| | __|  | . ` |/ _ \ \/ / | | / __| 
         | |__| |      | | | | | (_| | | | \__ \ | |_   | |\  |  __/>  <| |_| \__ \ 
          \___\_\      |_| |_|  \__,_|_| |_|___/_|\__|  |_| \_|\___/_/\_\\__,_|___/ 
                                                                                    
   MULTIMODAL URBAN TRANSPORTATION DIGITAL TWIN & SIMULATION ENGINE (SIH)
=============================================================================
""")

def main():
    banner()
    parser = argparse.ArgumentParser(description="Q-Transit Nexus Master Demo CLI")
    parser.add_argument(
        "--scenario",
        choices=["normal", "accident", "congestion", "emergency", "recovery"],
        default="accident",
        help="Scenario to execute (default: accident)"
    )
    parser.add_argument("--all", action="store_true", help="Run all 5 scenarios in sequence")
    parser.add_argument("--gui", action="store_true", help="Run with visual SUMO-GUI")
    parser.add_argument("--steps", type=int, default=300, help="Simulation duration (default: 300s)")
    args = parser.parse_args()

    if args.all:
        print("[DEMO SUITE] Executing full multimodal benchmark suite...")
        time.sleep(1)
        print("\n--- 1/5: NORMAL BASELINE ---")
        run_normal_scenario(steps=args.steps, gui=args.gui)
        print("\n--- 2/5: ACCIDENT & DYNAMIC REROUTE (AT-DQPSO) ---")
        run_accident_benchmark(steps=args.steps, gui=args.gui)
        print("\n--- 3/5: EMERGENCY PREEMPTION (GREEN WAVE) ---")
        run_emergency_benchmark(steps=args.steps, gui=args.gui)
        print("\n--- 4/5: CONGESTION & ADAPTIVE SIGNALS ---")
        run_congestion_benchmark(steps=args.steps, gui=args.gui)
        print("\n--- 5/5: INCIDENT CLEARANCE & RECOVERY ---")
        run_recovery_scenario(steps=args.steps + 50, gui=args.gui)
        print("\n[ALL COMPLETED] Full benchmark suite executed successfully!")
        print("Summary reports available in: benchmark_results/")
        return

    if args.scenario == "normal":
        run_normal_scenario(steps=args.steps, gui=args.gui)
    elif args.scenario == "accident":
        run_accident_benchmark(steps=args.steps, gui=args.gui)
    elif args.scenario == "congestion":
        run_congestion_benchmark(steps=args.steps, gui=args.gui)
    elif args.scenario == "emergency":
        run_emergency_benchmark(steps=args.steps, gui=args.gui)
    elif args.scenario == "recovery":
        run_recovery_scenario(steps=args.steps, gui=args.gui)

if __name__ == "__main__":
    main()
