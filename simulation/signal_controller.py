"""
Module: simulation.signal_controller
Description:
    Manages traffic signal timing plans across controlled junctions (TL_J11, TL_J21).
    Supports standard coordinated cycles, bus transit priority extensions,
    and Emergency Vehicle Preemption (EVP / Green Wave).

What it reads:
    - TraCI traffic light states and approaching vehicle telemetry (distance, type, priority)

What it writes:
    - TraCI traffic light phase modifications (traci.trafficlight.setPhase, setPhaseDuration)
    - Preemption audit logs

Dependencies:
    - Depends on: traci_controller.py
    - Depended on by: scenario_manager.py, AT-DQPSO signal optimization loop
"""

from typing import Dict, Any, Optional, List

# Phase mappings for prototype junctions J11 and J21:
# Phase 0: East-West Arterial Green (duration ~31s)
# Phase 1: East-West Yellow (duration 4s)
# Phase 2: North-South Cross-Street Green (duration ~25s)
# Phase 3: North-South Yellow (duration 4s)
PHASE_EW_GREEN = 0
PHASE_EW_YELLOW = 1
PHASE_NS_GREEN = 2
PHASE_NS_YELLOW = 3

class SignalController:
    """
    Traffic signal manager supporting fixed-time coordination,
    bus green extension, and Emergency Vehicle Preemption (EVP).
    """

    def __init__(self, controller):
        self.controller = controller
        self.preempted_signals: Dict[str, Dict[str, Any]] = {}
        self.evp_history: List[Dict[str, Any]] = []

    def set_phase(self, tl_id: str, phase_index: int, duration: Optional[float] = None):
        """Forces a specific phase index on a traffic light."""
        traci = self.controller.traci
        if not self.controller.is_connected or not traci:
            return
        try:
            traci.trafficlight.setPhase(tl_id, phase_index)
            if duration is not None:
                traci.trafficlight.setPhaseDuration(tl_id, duration)
        except Exception:
            pass

    def check_and_apply_evp(self, emergency_veh_id: str = "ambulance_MED01", detection_range_m: float = 160.0):
        """
        Scans for the emergency vehicle approaching signalized junctions.
        If within detection_range_m, preempts the junction signal to ensure an uninterrupted Green Wave.
        """
        traci = self.controller.traci
        sim_time = self.controller.get_time()

        if not self.controller.is_connected or not traci:
            return

        try:
            active_vehs = traci.vehicle.getIDList()
            if emergency_veh_id not in active_vehs:
                # If ambulance is not in network, release any active preemptions
                for tl_id in list(self.preempted_signals.keys()):
                    self.release_preemption(tl_id)
                return

            cur_edge = traci.vehicle.getRoadID(emergency_veh_id)
            cur_pos = traci.vehicle.getLanePosition(emergency_veh_id)

            # Junction J11 incoming edges:
            # - Eastbound: E_01_11 (length 300m)
            # - Westbound: E_21_11 (length 300m)
            # - Northbound: E_10_11 (length 300m)
            # - Southbound: E_12_11 (length 300m)

            # Check J11 approach
            if cur_edge == "E_01_11" or cur_edge == "E_21_11":
                edge_len = 300.0
                dist_to_junc = edge_len - cur_pos
                if dist_to_junc <= detection_range_m:
                    self.trigger_emergency_preemption("TL_J11", green_phase_index=PHASE_EW_GREEN, duration=35.0)

            elif cur_edge == "E_10_11" or cur_edge == "E_12_11":
                edge_len = 300.0
                dist_to_junc = edge_len - cur_pos
                if dist_to_junc <= detection_range_m:
                    self.trigger_emergency_preemption("TL_J11", green_phase_index=PHASE_NS_GREEN, duration=35.0)

            # Check J21 approach
            if cur_edge == "E_11_21" or cur_edge == "E_31_21":
                edge_len = 300.0
                dist_to_junc = edge_len - cur_pos
                if dist_to_junc <= detection_range_m:
                    self.trigger_emergency_preemption("TL_J21", green_phase_index=PHASE_EW_GREEN, duration=35.0)

            elif cur_edge == "E_20_21" or cur_edge == "E_22_21":
                edge_len = 300.0
                dist_to_junc = edge_len - cur_pos
                if dist_to_junc <= detection_range_m:
                    self.trigger_emergency_preemption("TL_J21", green_phase_index=PHASE_NS_GREEN, duration=35.0)

            # If vehicle has passed past junction, release preemption
            for tl_id, data in list(self.preempted_signals.items()):
                if sim_time - data["preempt_time"] > data["duration"]:
                    self.release_preemption(tl_id)

        except Exception:
            pass

    def trigger_emergency_preemption(self, tl_id: str, green_phase_index: int = PHASE_EW_GREEN, duration: float = 30.0):
        """Preempts normal cycle to grant green wave to approaching emergency vehicle."""
        traci = self.controller.traci
        sim_time = self.controller.get_time()

        if tl_id in self.preempted_signals:
            return  # Already preempted

        try:
            self.preempted_signals[tl_id] = {
                "preempt_time": sim_time,
                "duration": duration,
                "target_phase": green_phase_index
            }
            traci.trafficlight.setPhase(tl_id, green_phase_index)
            traci.trafficlight.setPhaseDuration(tl_id, duration)

            record = {
                "time": sim_time,
                "junction_id": tl_id,
                "phase": green_phase_index,
                "duration": duration,
                "status": "PREEMPTED_GREEN_WAVE"
            }
            self.evp_history.append(record)
        except Exception:
            pass

    def release_preemption(self, tl_id: str):
        """Restores default traffic signal cycle after emergency vehicle clears."""
        if tl_id in self.preempted_signals:
            del self.preempted_signals[tl_id]

    def apply_bus_priority_extension(self, bus_id: str, extension_s: float = 8.0):
        """Extends current green phase if an approaching bus is within 50m of intersection."""
        traci = self.controller.traci
        if not self.controller.is_connected or not traci:
            return
        try:
            cur_edge = traci.vehicle.getRoadID(bus_id)
            cur_pos = traci.vehicle.getLanePosition(bus_id)
            dist_to_end = 300.0 - cur_pos

            if dist_to_end <= 50.0:
                tl_id = "TL_J11" if "11" in cur_edge else ("TL_J21" if "21" in cur_edge else None)
                if tl_id and tl_id not in self.preempted_signals:
                    cur_phase = traci.trafficlight.getPhase(tl_id)
                    if cur_phase in (PHASE_EW_GREEN, PHASE_NS_GREEN):
                        cur_dur = traci.trafficlight.getPhaseDuration(tl_id)
                        traci.trafficlight.setPhaseDuration(tl_id, cur_dur + extension_s)
        except Exception:
            pass
