"""
Module: simulation.state_manager
Description:
    Defines and constructs the canonical Digital-Twin State Object for Q-Transit Nexus.
    Extracts real-time telemetry from TraCI across road edges, private vehicles, public
    transit buses, traffic light signals, active incidents, and electric vehicles.

What it reads:
    - TraCI simulation state (vehicles, edges, signals, stops)
    - Incident states tracked by scenario_manager
    - EV battery telemetry from ev_manager

What it writes:
    - Canonical Digital-Twin State dictionary (JSON-serializable contract v1.0.0)

Dependencies:
    - Depends on: traci_controller.py
    - Depended on by: AT-DQPSO optimizer, GNN traffic prediction layer, metrics.py, dashboard API
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

SCHEMA_VERSION = "1.0.0"

@dataclass
class EdgeState:
    edge_id: str
    vehicle_count: int
    mean_speed: float          # m/s
    occupancy: float           # 0.0 to 1.0 (percentage of road occupied)
    waiting_time: float        # total accumulated waiting time on edge (s)
    travel_time: float         # estimated travel time over edge (s)

@dataclass
class VehicleState:
    vehicle_id: str
    type: str                  # passenger, delivery, bus, emergency, car_ev
    current_edge: str
    lane_index: int
    speed: float               # m/s
    position: float            # distance along current edge (m)
    route: List[str]           # sequence of edge IDs
    waiting_time: float        # seconds stopped
    travel_time: float         # total active travel time (s)
    priority: str              # HIGH, NORMAL, LOW
    is_rerouted: bool = False

@dataclass
class BusState:
    vehicle_id: str
    line: str
    current_edge: str
    current_stop: Optional[str]
    delay: float               # deviation from schedule (seconds)
    passenger_load: int        # number of onboard passengers
    speed: float               # m/s
    waiting_time: float

@dataclass
class SignalState:
    junction_id: str
    current_phase: int
    phase_duration: float
    state_string: str          # e.g. "GGgrrrGGgrrr"
    is_priority_preempted: bool

@dataclass
class IncidentState:
    incident_id: str
    edge_id: str
    incident_type: str         # accident, road_closure, traffic_surge
    severity: float            # 0.0 (cleared) to 1.0 (total blockage)
    start_time: float
    is_active: bool

@dataclass
class EVState:
    vehicle_id: str
    battery_soc: float         # 0.0 to 100.0 %
    energy_consumed_kwh: float
    is_charging: bool
    charger_id: Optional[str]

class DigitalTwinStateManager:
    """
    Constructs and verifies the canonical Digital-Twin State Object at any simulation step.
    Contract Schema Version: 1.0.0
    """

    def __init__(self, controller, ev_mgr=None):
        self.controller = controller
        self.ev_mgr = ev_mgr
        self.active_incidents: List[IncidentState] = []

    def set_ev_manager(self, ev_mgr):
        self.ev_mgr = ev_mgr

    def register_incident(self, incident: IncidentState):
        self.active_incidents.append(incident)

    def clear_incidents(self):
        self.active_incidents.clear()

    def get_state(self) -> Dict[str, Any]:
        """
        Polls TraCI and returns the canonical Digital-Twin state dictionary.
        Strict Schema Contract v1.0.0:
        {
            "schema_version": "1.0.0",
            "simulation_time": float,
            "edges": [...],
            "vehicles": [...],
            "buses": [...],
            "signals": [...],
            "incidents": [...],
            "evs": [...]
        }
        """
        traci = self.controller.traci
        sim_time = self.controller.get_time()

        edges_list: List[Dict[str, Any]] = []
        vehicles_list: List[Dict[str, Any]] = []
        buses_list: List[Dict[str, Any]] = []
        signals_list: List[Dict[str, Any]] = []
        evs_list: List[Dict[str, Any]] = []

        if self.controller.is_connected and traci:
            # 1. Edge States
            try:
                for eid in traci.edge.getIDList():
                    if eid.startswith(":"):
                        continue
                    edge_data = EdgeState(
                        edge_id=eid,
                        vehicle_count=traci.edge.getLastStepVehicleNumber(eid),
                        mean_speed=round(float(traci.edge.getLastStepMeanSpeed(eid)), 2),
                        occupancy=round(float(traci.edge.getLastStepOccupancy(eid)), 3),
                        waiting_time=round(float(traci.edge.getWaitingTime(eid)), 1),
                        travel_time=round(float(traci.edge.getAdaptedTraveltime(eid, sim_time)), 1)
                    )
                    edges_list.append(asdict(edge_data))
            except Exception:
                pass

            # 2. Vehicles & Buses
            try:
                for vid in traci.vehicle.getIDList():
                    v_type = traci.vehicle.getTypeID(vid)
                    cur_edge = traci.vehicle.getRoadID(vid)
                    lane_idx = traci.vehicle.getLaneIndex(vid)
                    speed = round(float(traci.vehicle.getSpeed(vid)), 2)
                    pos = round(float(traci.vehicle.getLanePosition(vid)), 2)
                    route = list(traci.vehicle.getRoute(vid))
                    waiting = round(float(traci.vehicle.getWaitingTime(vid)), 1)
                    depart_time = float(traci.vehicle.getDeparture(vid))
                    travel_time = round(max(0.0, sim_time - depart_time), 1)

                    try:
                        priority = traci.vehicle.getParameter(vid, "priority") or "NORMAL"
                    except Exception:
                        priority = "NORMAL"

                    try:
                        rerouted = traci.vehicle.getParameter(vid, "is_rerouted") == "true"
                    except Exception:
                        rerouted = False

                    if "bus" in v_type.lower():
                        try:
                            line = traci.vehicle.getLine(vid) or "B101"
                            pload = int(traci.vehicle.getParameter(vid, "passenger_load") or "25")
                            delay = round(float(traci.vehicle.getParameter(vid, "scheduled_delay") or "0.0"), 1)
                        except Exception:
                            line, pload, delay = "B101", 25, 0.0

                        b_state = BusState(
                            vehicle_id=vid,
                            line=line,
                            current_edge=cur_edge,
                            current_stop=None,
                            delay=delay,
                            passenger_load=pload,
                            speed=speed,
                            waiting_time=waiting
                        )
                        buses_list.append(asdict(b_state))
                    else:
                        v_state = VehicleState(
                            vehicle_id=vid,
                            type=v_type,
                            current_edge=cur_edge,
                            lane_index=lane_idx,
                            speed=speed,
                            position=pos,
                            route=route,
                            waiting_time=waiting,
                            travel_time=travel_time,
                            priority=priority,
                            is_rerouted=rerouted
                        )
                        vehicles_list.append(asdict(v_state))
            except Exception:
                pass

            # 3. Traffic Signals
            try:
                for tlid in traci.trafficlight.getIDList():
                    s_state = SignalState(
                        junction_id=tlid,
                        current_phase=int(traci.trafficlight.getPhase(tlid)),
                        phase_duration=float(traci.trafficlight.getPhaseDuration(tlid)),
                        state_string=str(traci.trafficlight.getRedYellowGreenState(tlid)),
                        is_priority_preempted=False
                    )
                    signals_list.append(asdict(s_state))
            except Exception:
                pass

            # 4. EV States (from EVManager if present)
            if self.ev_mgr:
                evs_list = [asdict(ev) for ev in self.ev_mgr.get_all_ev_states()]

        return {
            "schema_version": SCHEMA_VERSION,
            "simulation_time": sim_time,
            "edges": edges_list,
            "vehicles": vehicles_list,
            "buses": buses_list,
            "signals": signals_list,
            "incidents": [asdict(inc) for inc in self.active_incidents],
            "evs": evs_list
        }

    def export_snapshot_json(self, filepath: str):
        """Dumps current state snapshot to a JSON file."""
        state = self.get_state()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
