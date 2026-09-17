"""
Module: simulation.scenario_manager
Description:
    Orchestrates reproducible dynamic scenarios (normal baseline, accident with rerouting,
    congestion surge, emergency vehicle preemption, and incident recovery).
    Provides the primary interface for demo runners, automated benchmarking,
    and external dashboard triggers.

What it reads:
    - Scenario configs (edge, timing, duration, severity)
    - Subsystem controllers (TraCI, StateManager, RouteApplier, SignalController, Metrics, EVManager)

What it writes:
    - Active incident registrations, TraCI edge modifications, dynamic reroutes, EVP signal triggers
    - Run execution summaries and BEFORE/AFTER metrics benchmarks

Dependencies:
    - Depends on: traci_controller, state_manager, route_applier, signal_controller, metrics, ev_manager
    - Depended on by: simulation/scenarios/*.py, demo.py, dashboard API
"""

import time
from typing import Dict, Any, Optional, Callable
from simulation.state_manager import IncidentState

class ScenarioManager:
    """
    Coordinates dynamic events and executes reproducible end-to-end simulation scenarios.
    """

    def __init__(self, controller, state_mgr, route_applier, signal_ctrl, metrics_col, ev_mgr=None):
        self.controller = controller
        self.state_mgr = state_mgr
        self.route_applier = route_applier
        self.signal_ctrl = signal_ctrl
        self.metrics_col = metrics_col
        self.ev_mgr = ev_mgr
        self.active_scenarios: Dict[str, Dict[str, Any]] = {}
        self._blocked_edges: Dict[str, float] = {}

    def trigger_accident(self, edge_id: str = "E_11_21", duration_s: float = 300.0, severity: float = 1.0) -> Dict[str, Any]:
        """
        Simulates an accident blocking the designated edge.
        Restricts edge max speed to 0.1 m/s to induce physical bottleneck and spillover.
        """
        traci = self.controller.traci
        sim_time = self.controller.get_time()

        if self.controller.is_connected and traci:
            try:
                # Save original speed
                orig_spd = traci.edge.getLastStepMeanSpeed(edge_id)
                if orig_spd <= 0:
                    orig_spd = 16.67
                self._blocked_edges[edge_id] = orig_spd
                # Force bottleneck
                traci.edge.setMaxSpeed(edge_id, 0.1)
            except Exception:
                self._blocked_edges[edge_id] = 16.67

        incident = IncidentState(
            incident_id=f"INC_ACC_{int(sim_time)}",
            edge_id=edge_id,
            incident_type="accident",
            severity=severity,
            start_time=sim_time,
            is_active=True
        )
        self.state_mgr.register_incident(incident)

        event = {
            "type": "accident",
            "edge_id": edge_id,
            "start_time": sim_time,
            "duration": duration_s,
            "severity": severity,
            "status": "ACTIVE"
        }
        self.active_scenarios["accident"] = event
        return event

    def clear_accident(self, edge_id: str = "E_11_21") -> Dict[str, Any]:
        """Restores edge speed limit to nominal and clears the incident state."""
        traci = self.controller.traci
        orig_speed = self._blocked_edges.get(edge_id, 16.67)

        if self.controller.is_connected and traci:
            try:
                traci.edge.setMaxSpeed(edge_id, orig_speed)
            except Exception:
                pass

        self.state_mgr.clear_incidents()
        if "accident" in self.active_scenarios:
            self.active_scenarios["accident"]["status"] = "CLEARED"

        return {"status": "CLEARED", "edge_id": edge_id, "restored_speed": orig_speed}

    def step_cycle(self, enable_evp: bool = True, enable_ev: bool = True):
        """Advances simulation by 1 step and coordinates all sub-controllers."""
        sim_time = self.controller.step()

        # 1. Update EV battery models
        if enable_ev and self.ev_mgr:
            self.ev_mgr.step_ev_models(dt_seconds=1.0)

        # 2. Check Emergency Vehicle Preemption (EVP)
        if enable_evp and self.signal_ctrl:
            self.signal_ctrl.check_and_apply_evp(emergency_veh_id="ambulance_MED01")

        # 3. Capture time-series metrics
        if self.metrics_col:
            self.metrics_col.capture_snapshot()

        return sim_time

    def run_simulation_loop(self, total_steps: int = 300,
                            step_hook: Optional[Callable[[float], None]] = None,
                            enable_evp: bool = True,
                            enable_ev: bool = True) -> Any:
        """
        Executes a controlled simulation loop for total_steps.
        step_hook(sim_time) is invoked each step for custom dynamic event injection.
        """
        for _ in range(total_steps):
            t = self.step_cycle(enable_evp=enable_evp, enable_ev=enable_ev)
            if step_hook:
                step_hook(t)
        return self.metrics_col.get_summary()
