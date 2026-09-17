"""
Headless and GUI Execution Runner for Q-Transit Nexus Simulation Layer.

Usage:
    python run_simulation.py                # Run headless for 500 steps
    python run_simulation.py --gui          # Run in SUMO-GUI
    python run_simulation.py --steps 1000   # Run for 1000 steps
"""

import os
import sys
import argparse
import shutil
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(PROJECT_ROOT, "simulation", "sumo_config", "simulation.sumocfg")

def find_sumo_binary(gui: bool = False) -> str:
    target = "sumo-gui" if gui else "sumo"
    # 1. Search PATH
    path = shutil.which(target)
    if path:
        return path

    # 2. Search SUMO_HOME
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        cand = os.path.join(sumo_home, "bin", f"{target}.exe" if os.name == "nt" else target)
        if os.path.isfile(cand):
            return cand

    # 3. Check common Windows installation paths
    if os.name == "nt":
        common_paths = [
            r"C:\Program Files (x86)\Eclipse\Sumo\bin",
            r"C:\Program Files\Eclipse\Sumo\bin",
            r"C:\Sumo\bin"
        ]
        for cp in common_paths:
            cand = os.path.join(cp, f"{target}.exe")
            if os.path.isfile(cand):
                return cand

    return ""

def run_simulation(gui: bool = False, steps: int = 500):
    print("=" * 65)
    print("Q-TRANSIT NEXUS: Digital Twin & Simulation Runner (Stage 1)")
    print("=" * 65)
    print(f"Configuration : {CONFIG_FILE}")
    print(f"Mode          : {'GUI' if gui else 'Headless (CLI)'}")
    print(f"Target steps  : {steps} seconds")
    print("-" * 65)

    if not os.path.isfile(CONFIG_FILE):
        print(f"[ERROR] Configuration file not found: {CONFIG_FILE}")
        sys.exit(1)

    sumo_bin = find_sumo_binary(gui)
    if not sumo_bin:
        print(f"[ERROR] Could not find '{'sumo-gui' if gui else 'sumo'}' executable.")
        print("Please ensure Eclipse SUMO is installed and SUMO_HOME is set in your environment.")
        print("Download SUMO: https://eclipse.dev/sumo/")
        sys.exit(1)

    print(f"[INFO] Using SUMO binary: {sumo_bin}")
    cmd = [
        sumo_bin,
        "-c", CONFIG_FILE,
        "--end", str(steps),
        "--duration-log.statistics", "true",
        "--no-step-log", "true",
        "--waiting-time-memory", "1000",
        "--time-to-teleport", "120"
    ]

    if gui:
        cmd.extend(["--start", "true"])

    print("[INFO] Launching simulation...")
    try:
        res = subprocess.run(cmd, capture_output=False, text=True)
        if res.returncode == 0:
            print("-" * 65)
            print("[SUCCESS] Simulation completed successfully!")
            print("=" * 65)
        else:
            print("-" * 65)
            print(f"[ERROR] Simulation exited with return code {res.returncode}")
    except Exception as exc:
        print(f"[ERROR] Execution failed: {exc}")

def main():
    parser = argparse.ArgumentParser(description="Q-Transit Nexus Simulation Runner")
    parser.add_argument("--gui", action="store_true", help="Launch in SUMO-GUI")
    parser.add_argument("--steps", type=int, default=500, help="Simulation duration in seconds (default: 500)")
    args = parser.parse_args()

    run_simulation(gui=args.gui, steps=args.steps)

if __name__ == "__main__":
    main()
