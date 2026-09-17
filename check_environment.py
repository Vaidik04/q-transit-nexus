"""
Pre-flight check script for Q-Transit Nexus Simulation Environment.
Verifies Python version, SUMO binaries (sumo, sumo-gui, netconvert),
SUMO_HOME environment variable, and required Python packages (traci, sumolib).
"""

import sys
import os
import shutil
import subprocess

def check_env():
    print("=" * 60)
    print("Q-TRANSIT NEXUS: Simulation Environment Pre-Flight Check")
    print("=" * 60)

    # 1. Python Check
    print(f"[OK] Python version: {sys.version.split()[0]} ({sys.executable})")

    # 2. SUMO_HOME Check
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        print(f"[OK] SUMO_HOME is set: {sumo_home}")
        if os.path.isdir(sumo_home):
            print(f"     -> Directory exists: True")
        else:
            print(f"     -> [WARNING] Directory does not exist: {sumo_home}")
    else:
        print("[WARNING] SUMO_HOME environment variable is NOT set.")

    # 3. Executables Check
    binaries = ["sumo", "sumo-gui", "netconvert", "netgenerate"]
    found_binaries = {}
    for b in binaries:
        path = shutil.which(b)
        if not path and sumo_home:
            candidate = os.path.join(sumo_home, "bin", f"{b}.exe")
            if os.path.isfile(candidate):
                path = candidate
        found_binaries[b] = path
        if path:
            print(f"[OK] Found '{b}': {path}")
        else:
            print(f"[INFO] '{b}' not found in PATH or SUMO_HOME/bin")

    # 4. Check SUMO Version if available
    sumo_bin = found_binaries.get("sumo")
    if sumo_bin:
        try:
            res = subprocess.run([sumo_bin, "--version"], capture_output=True, text=True, timeout=5)
            first_line = res.stdout.strip().split("\n")[0] if res.stdout else "Unknown version"
            print(f"[OK] SUMO CLI version info: {first_line}")
        except Exception as e:
            print(f"[WARNING] Could not execute {sumo_bin} --version: {e}")

    # 5. Check Python Packages (traci, sumolib)
    packages = ["traci", "sumolib"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"[OK] Python package '{pkg}' is installed.")
        except ImportError:
            # Check if tools in SUMO_HOME can be imported
            if sumo_home:
                tools_dir = os.path.join(sumo_home, "tools")
                if tools_dir not in sys.path:
                    sys.path.append(tools_dir)
                try:
                    __import__(pkg)
                    print(f"[OK] Python package '{pkg}' found via SUMO_HOME/tools.")
                    continue
                except ImportError:
                    pass
            print(f"[INFO] Python package '{pkg}' is NOT installed in current Python environment.")

    print("=" * 60)
    print("Pre-flight check completed.")
    print("=" * 60)

if __name__ == "__main__":
    check_env()
