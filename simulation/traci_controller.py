"""
Module: simulation.traci_controller
Description:
    Manages the lifecycle of the TraCI connection to Eclipse SUMO. Provides clean primitives
    to start SUMO (headless or GUI), step the simulation forward, extract raw vehicle
    and edge telemetry, and cleanly disconnect without leaking child processes.

What it reads:
    - SUMO configuration path (simulation.sumocfg)
    - Live TraCI vehicle, edge, and traffic light subscriptions

What it writes:
    - TraCI control commands (simulationStep, vehicle route injections, signal phase updates)
    - Raw simulation state dictionaries

Dependencies:
    - Depends on: traci, sumolib (or SUMO binaries via SUMO_HOME)
    - Depended on by: state_manager.py, route_applier.py, signal_controller.py, scenario_manager.py
"""

import os
import sys
import shutil
import atexit
from typing import Optional, Dict, Any, List

def init_sumo_env() -> str:
    """Ensures SUMO_HOME is set and tools directory is in sys.path."""
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home and os.name == "nt":
        common_paths = [
            r"C:\Program Files (x86)\Eclipse\Sumo",
            r"C:\Program Files\Eclipse\Sumo",
            r"C:\Sumo"
        ]
        for cp in common_paths:
            if os.path.isdir(cp):
                sumo_home = cp
                os.environ["SUMO_HOME"] = cp
                break

    if sumo_home and os.path.isdir(sumo_home):
        tools_path = os.path.join(sumo_home, "tools")
        if tools_path not in sys.path:
            sys.path.append(tools_path)
        return sumo_home
    return ""

def find_sumo_binary(gui: bool = False) -> str:
    """Finds sumo or sumo-gui executable in PATH, SUMO_HOME, or standard directories."""
    target = "sumo-gui" if gui else "sumo"
    path = shutil.which(target)
    if path:
        return path

    sumo_home = init_sumo_env()
    if sumo_home:
        cand = os.path.join(sumo_home, "bin", f"{target}.exe" if os.name == "nt" else target)
        if os.path.isfile(cand):
            return cand

    return ""

class TraCIController:
    """
    Encapsulates SUMO TraCI process management, connection lifecycle,
    and high-level simulation stepping.
    """

    def __init__(self, config_file: str, gui: bool = False, step_length: float = 1.0, port: Optional[int] = None):
        self.config_file = os.path.abspath(config_file)
        self.gui = gui
        self.step_length = step_length
        self.port = port
        self.is_connected = False
        self._traci = None
        self._label = f"q_nexus_{os.getpid()}"
        init_sumo_env()
        # Register graceful exit handler
        atexit.register(self.close)

    def start(self, additional_args: Optional[List[str]] = None):
        """Starts the SUMO process and initializes the TraCI connection."""
        if self.is_connected:
            return self._traci

        try:
            import traci
            self._traci = traci
        except ImportError:
            raise RuntimeError(
                "TraCI package not found. Ensure SUMO_HOME is configured or run 'pip install traci sumolib'."
            )

        binary = find_sumo_binary(self.gui)
        if not binary:
            binary = "sumo-gui" if self.gui else "sumo"

        cmd = [
            binary,
            "-c", self.config_file,
            "--step-length", str(self.step_length),
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "120"
        ]
        if self.gui:
            cmd.extend(["--start", "true"])
        if additional_args:
            cmd.extend(additional_args)

        try:
            if self.port:
                self._traci.start(cmd, port=self.port, label=self._label)
            else:
                self._traci.start(cmd, label=self._label)
            self.is_connected = True
        except Exception as e:
            self.is_connected = False
            raise RuntimeError(f"Failed to connect TraCI to SUMO ({binary}): {e}")

        return self._traci

    def step(self) -> float:
        """Advances the simulation by one step and returns current simulation time in seconds."""
        if not self.is_connected or not self._traci:
            raise RuntimeError("TraCI is not running. Call start() first.")
        try:
            self._traci.simulationStep()
            return self._traci.simulation.getTime()
        except Exception as e:
            self.is_connected = False
            raise RuntimeError(f"TraCI simulation step failed: {e}")

    def get_time(self) -> float:
        """Returns the current simulation time."""
        if self.is_connected and self._traci:
            try:
                return self._traci.simulation.getTime()
            except Exception:
                pass
        return 0.0

    def is_running(self) -> bool:
        """Checks if the simulation is active and has remaining vehicles/steps."""
        if not self.is_connected or not self._traci:
            return False
        try:
            return self._traci.simulation.getMinExpectedNumber() > 0
        except Exception:
            return False

    def close(self):
        """Cleanly terminates the TraCI session and stops the SUMO process."""
        if self.is_connected and self._traci:
            try:
                self._traci.close()
            except Exception:
                pass
            finally:
                self.is_connected = False

    @property
    def traci(self):
        """Direct handle to the underlying traci module."""
        return self._traci
