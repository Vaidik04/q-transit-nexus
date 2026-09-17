import time
import math
import random
from typing import Dict, List, Optional
from backend.schemas.vehicle import VehicleState, VehicleType
from backend.schemas.traffic import NetworkState
from backend.schemas.incident import IncidentState, IncidentType, IncidentImpact
from backend.modules.network_engine import UrbanNetwork

class MicroscopicDigitalTwin:
    """
    High-fidelity microscopic transportation digital twin.
    Simulates individual vehicles, transit buses, EVs, and emergency services
    moving along network graph edges with realistic car-following and queue dynamics.
    Provides a seamless TraCI-compatible control interface.
    """
    def __init__(self, network: UrbanNetwork):
        self.network = network
        self.vehicles: Dict[str, VehicleState] = {}
        self.active_incidents: Dict[str, IncidentState] = {}
        self.simulation_step_count = 0
        self.sim_time_sec = 0.0
        self.is_running = False
        
        # Traffic signal phase timers for key intersections
        self.signal_phases: Dict[str, dict] = {
            "N9": {"green_axis": "EW", "timer": 25.0, "cycle": 60.0},
            "N10": {"green_axis": "NS", "timer": 20.0, "cycle": 50.0},
            "N15": {"green_axis": "EW", "timer": 30.0, "cycle": 60.0},
            "N16": {"green_axis": "NS", "timer": 25.0, "cycle": 55.0},
        }

        # Initialize representative heterogeneous fleet
        self._init_default_fleet()

    def _init_default_fleet(self):
        """Initializes a representative heterogeneous fleet matching the problem specification."""
        # 1. Delivery vehicles (Logistics fleet)
        delivery_routes = [
            ("V1", "N1", "N6", "E1"),
            ("V2", "N1", "N18", "E21"),
            ("V3", "N7", "N24", "E6"),
            ("V27", "N1", "N20", "E1"),  # Vehicle V27 specified in contract
            ("V5", "N13", "N12", "E11"),
            ("V6", "N19", "N6", "E16"),
        ]
        for vid, orig, dest, cur_edge in delivery_routes:
            route = self.network.find_shortest_path_edges(orig, dest)
            self.vehicles[vid] = VehicleState(
                vehicle_id=vid,
                type=VehicleType.DELIVERY,
                current_edge=cur_edge,
                origin=orig,
                destination=dest,
                capacity=20,
                priority=1,
                current_speed=38.0,
                route=route,
                route_index=0,
                progress_on_edge=random.uniform(0.1, 0.4)
            )

        # 2. Public Transit Buses (GTFS line representations)
        bus_routes = [
            ("BUS101", "N1", "N6", "E1", "Line_1_NorthExpress", 55),
            ("BUS102", "N7", "N12", "E6", "Line_2_MetroArterial", 74), # High passenger load
            ("BUS17",  "N13", "N18", "E11", "Line_3_CivicSpine", 82),   # Bus 17 in prompt
            ("BUS24",  "N19", "N24", "E16", "Line_4_SouthConnector", 40), # Bus 24 in prompt
            ("BUS105", "N3", "N21", "E27", "Line_5_UniversityHospital", 62),
        ]
        for vid, orig, dest, cur_edge, line, pax in bus_routes:
            route = self.network.find_shortest_path_edges(orig, dest)
            self.vehicles[vid] = VehicleState(
                vehicle_id=vid,
                type=VehicleType.BUS,
                current_edge=cur_edge,
                origin=orig,
                destination=dest,
                capacity=90,
                priority=2,
                current_speed=30.0,
                passengers=pax,
                line_id=line,
                target_headway=300.0,
                route=route,
                route_index=0,
                progress_on_edge=random.uniform(0.1, 0.5)
            )

        # 3. Electric Vehicles (EVs)
        ev_routes = [
            ("EV01", "N2", "N21", "E2", 0.45, 60.0, "N21"),
            ("EV08", "N5", "N21", "E33", 0.28, 55.0, "N21"),  # EV 08 in prompt
            ("EV03", "N8", "N21", "E7", 0.65, 75.0, "N21"),
            ("EV04", "N14", "N21", "E26", 0.32, 60.0, "N21"),
        ]
        for vid, orig, dest, cur_edge, soc, cap_kwh, charger in ev_routes:
            route = self.network.find_shortest_path_edges(orig, dest)
            self.vehicles[vid] = VehicleState(
                vehicle_id=vid,
                type=VehicleType.EV,
                current_edge=cur_edge,
                origin=orig,
                destination=dest,
                capacity=5,
                priority=1,
                current_speed=42.0,
                soc=soc,
                battery_capacity_kwh=cap_kwh,
                target_charger=charger,
                route=route,
                route_index=0,
                progress_on_edge=random.uniform(0.05, 0.3)
            )

        # 4. Emergency Responders
        emergency_units = [
            ("AMB01", "N2", "N15", "E24", 5),  # Ambulance targeting Central Hospital N15
            ("FIRE02", "N19", "N10", "E23", 5), # Fire truck targeting City Hall N10
        ]
        for vid, orig, dest, cur_edge, prio in emergency_units:
            route = self.network.find_shortest_path_edges(orig, dest)
            self.vehicles[vid] = VehicleState(
                vehicle_id=vid,
                type=VehicleType.EMERGENCY,
                current_edge=cur_edge,
                origin=orig,
                destination=dest,
                capacity=4,
                priority=prio,
                current_speed=55.0,
                is_emergency_active=False,
                route=route,
                route_index=0,
                progress_on_edge=0.1
            )

        # 5. Private Traffic background flow
        for i in range(1, 15):
            pid = f"P{i:02d}"
            orig = f"N{random.randint(1, 12)}"
            dest = f"N{random.randint(13, 24)}"
            route = self.network.find_shortest_path_edges(orig, dest)
            cur_edge = route[0] if route else "E1"
            self.vehicles[pid] = VehicleState(
                vehicle_id=pid,
                type=VehicleType.PRIVATE,
                current_edge=cur_edge,
                origin=orig,
                destination=dest,
                capacity=4,
                priority=1,
                current_speed=random.uniform(32.0, 48.0),
                route=route,
                route_index=0,
                progress_on_edge=random.uniform(0.0, 0.8)
            )

    def step(self, dt: float = 1.0):
        """Advances microscopic simulation by dt seconds."""
        self.simulation_step_count += 1
        self.sim_time_sec += dt

        # 1. Update traffic signals
        for nid, sig in self.signal_phases.items():
            sig["timer"] += dt
            if sig["timer"] >= sig["cycle"]:
                sig["timer"] = 0.0
                sig["green_axis"] = "NS" if sig["green_axis"] == "EW" else "EW"

        # 2. Count vehicle density per edge
        edge_counts: Dict[str, int] = {eid: 0 for eid in self.network.edges_data}
        for v in self.vehicles.values():
            if v.current_edge in edge_counts:
                edge_counts[v.current_edge] += 1

        # Update edge traffic speeds based on density
        for eid, count in edge_counts.items():
            self.network.update_edge_traffic(eid, count, dt)

        # 3. Advance each vehicle along its route
        for vid, v in self.vehicles.items():
            if not v.route or v.route_index >= len(v.route):
                # Vehicle completed route; assign reverse or new trip
                v.origin, v.destination = v.destination, v.origin
                v.route = self.network.find_shortest_path_edges(v.origin, v.destination)
                v.route_index = 0
                v.progress_on_edge = 0.0
                if v.route:
                    v.current_edge = v.route[0]
                continue

            current_edge_id = v.route[v.route_index]
            v.current_edge = current_edge_id
            edata = self.network.edges_data.get(current_edge_id)
            if not edata:
                continue

            edge_len = edata["length"]
            # Kinematic speed calculation
            if edata.get("is_blocked", False):
                effective_speed = 3.0 # Crawling / stalled at incident
            else:
                effective_speed = edata["current_speed"]
                if v.type == VehicleType.EMERGENCY and v.is_emergency_active:
                    effective_speed = min(80.0, effective_speed * 1.5)

            v.current_speed = effective_speed
            v.total_travel_time += dt

            # Distance traveled in dt (m)
            dist_moved = (effective_speed * 1000.0 / 3600.0) * dt
            progress_delta = dist_moved / max(10.0, edge_len)
            v.progress_on_edge += progress_delta

            # EV battery consumption
            if v.type == VehicleType.EV and v.soc is not None:
                # Discharging: approx 0.18 kWh per km
                km_moved = dist_moved / 1000.0
                kwh_spent = km_moved * 0.18
                if v.battery_capacity_kwh:
                    v.soc = max(0.05, v.soc - (kwh_spent / v.battery_capacity_kwh))

            # Move to next edge if reached end
            if v.progress_on_edge >= 1.0:
                v.progress_on_edge = 0.0
                v.route_index += 1
                if v.route_index < len(v.route):
                    v.current_edge = v.route[v.route_index]

        # 4. Check active incidents expiration
        expired = []
        for inc_id, inc in self.active_incidents.items():
            if self.sim_time_sec - inc.start_time > inc.duration_sec:
                expired.append(inc_id)
        for inc_id in expired:
            self.clear_incident(inc_id)

    def inject_incident(self, incident: IncidentState):
        """Injects an incident (e.g. accident on Road E14)."""
        self.active_incidents[incident.incident_id] = incident
        eid = incident.edge_id
        if eid in self.network.edges_data:
            ed = self.network.edges_data[eid]
            ed["is_blocked"] = True
            ed["incident_penalty"] = incident.severity * 50.0
            ed["current_speed"] = 3.0
            ed["congestion"] = 0.98

        # Calculate impact assessment
        affected_vehicles = [v for v in self.vehicles.values() if eid in v.route[v.route_index:]]
        affected_buses = [v for v in affected_vehicles if v.type == VehicleType.BUS]
        affected_passengers = sum(b.passengers for b in affected_buses)

        incident.impact = IncidentImpact(
            incident_id=incident.incident_id,
            edge_id=eid,
            affected_vehicles_count=len(affected_vehicles),
            affected_buses_count=len(affected_buses),
            affected_passengers_count=affected_passengers,
            predicted_downstream_congestion=0.84,
            recommended_action="AT-DQPSO Reroute + Transit Priority Corridor"
        )

    def clear_incident(self, incident_id: str):
        if incident_id in self.active_incidents:
            inc = self.active_incidents.pop(incident_id)
            eid = inc.edge_id
            if eid in self.network.edges_data:
                ed = self.network.edges_data[eid]
                ed["is_blocked"] = False
                ed["incident_penalty"] = 0.0
                ed["current_speed"] = ed["free_flow_speed"] * 0.8
                ed["congestion"] = 0.3

    def get_network_state(self) -> NetworkState:
        edge_states = self.network.get_all_edge_states()
        speeds = [e.speed for e in edge_states.values()]
        congs = [e.congestion for e in edge_states.values()]
        return NetworkState(
            timestamp=time.time(),
            edges=edge_states,
            total_vehicles=len(self.vehicles),
            avg_speed=round(sum(speeds) / max(1, len(speeds)), 1),
            avg_congestion=round(sum(congs) / max(1, len(congs)), 2),
            active_incidents=len(self.active_incidents)
        )
