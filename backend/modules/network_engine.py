import math
from typing import Dict, List, Tuple, Optional
import networkx as nx
from backend.schemas.traffic import EdgeState

class UrbanNetwork:
    """
    Multimodal urban road network graph backed by NetworkX.
    Includes realistic node coordinates, edge capacities, free-flow speeds,
    and BPR (Bureau of Public Roads) travel time functions.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes_data: Dict[str, dict] = {}
        self.edges_data: Dict[str, dict] = {}
        self.edge_by_nodes: Dict[Tuple[str, str], str] = {}
        self._build_default_city_network()

    def _build_default_city_network(self):
        """
        Creates a realistic urban network with 24 nodes (N1..N24) representing
        a city center with main arterial corridors, ring roads, bridges, transit lanes,
        and residential collectors. Matches edge naming E1..E38.
        """
        # Node coordinates (x, y in km scale for display and distance)
        node_positions = {
            "N1": {"x": 100, "y": 100, "name": "West Depot / Logistics Hub"},
            "N2": {"x": 250, "y": 100, "name": "West Gate Junction"},
            "N3": {"x": 400, "y": 100, "name": "University District"},
            "N4": {"x": 550, "y": 100, "name": "North Transit Center"},
            "N5": {"x": 700, "y": 100, "name": "North Residential"},
            "N6": {"x": 850, "y": 100, "name": "Tech Park North"},
            
            "N7": {"x": 100, "y": 250, "name": "West Industrial Zone"},
            "N8": {"x": 250, "y": 250, "name": "Avenue Central West"},
            "N9": {"x": 400, "y": 250, "name": "Grand Metro Station"},
            "N10": {"x": 550, "y": 250, "name": "Civic Center / City Hall"},
            "N11": {"x": 700, "y": 250, "name": "Financial District South"},
            "N12": {"x": 850, "y": 250, "name": "East River Cross Point"},

            "N13": {"x": 100, "y": 400, "name": "Southwest Suburbs"},
            "N14": {"x": 250, "y": 400, "name": "Hospital Corridor Jct"},
            "N15": {"x": 400, "y": 400, "name": "Central Hospital & ER"},
            "N16": {"x": 550, "y": 400, "name": "Market Square"},
            "N17": {"x": 700, "y": 400, "name": "Commercial Boulevard"},
            "N18": {"x": 850, "y": 400, "name": "East Freight Terminal"},

            "N19": {"x": 100, "y": 550, "name": "South Port Depot"},
            "N20": {"x": 250, "y": 550, "name": "South Highway Interchange"},
            "N21": {"x": 400, "y": 550, "name": "South EV Fast-Charge Hub"},
            "N22": {"x": 550, "y": 550, "name": "Innovation Hub"},
            "N23": {"x": 700, "y": 550, "name": "East Residential Belt"},
            "N24": {"x": 850, "y": 550, "name": "East Gateway Junction"},
        }

        for nid, data in node_positions.items():
            self.nodes_data[nid] = data
            self.graph.add_node(nid, **data)

        # Edges definitions: (from_node, to_node, edge_id, length_m, free_flow_kmh, capacity_vph, is_transit_lane)
        edges_spec = [
            # Horizontal corridors Row 1
            ("N1", "N2", "E1", 450.0, 50.0, 1400.0, False),
            ("N2", "N3", "E2", 450.0, 50.0, 1400.0, False),
            ("N3", "N4", "E3", 450.0, 45.0, 1200.0, True),
            ("N4", "N5", "E4", 450.0, 50.0, 1400.0, False),
            ("N5", "N6", "E5", 450.0, 60.0, 1600.0, False),
            
            # Row 2
            ("N7", "N8", "E6", 450.0, 50.0, 1200.0, False),
            ("N8", "N9", "E7", 450.0, 40.0, 1500.0, True),
            ("N9", "N10", "E8", 450.0, 45.0, 1800.0, True),
            ("N10", "N11", "E9", 450.0, 50.0, 1600.0, False),
            ("N11", "N12", "E10", 450.0, 55.0, 1500.0, False),

            # Row 3 (Arterial spine)
            ("N13", "N14", "E11", 450.0, 50.0, 1300.0, False),
            ("N14", "N15", "E12", 450.0, 40.0, 1400.0, True),  # Hospital Access
            ("N15", "N16", "E13", 450.0, 35.0, 1200.0, False),
            ("N16", "N17", "E14", 420.0, 45.0, 1500.0, False), # Road 14 (E14) explicitly specified in prompt
            ("N17", "N18", "E15", 450.0, 55.0, 1600.0, False),

            # Row 4
            ("N19", "N20", "E16", 450.0, 60.0, 1800.0, False),
            ("N20", "N21", "E17", 450.0, 50.0, 1400.0, False),
            ("N21", "N22", "E18", 450.0, 45.0, 1200.0, False),
            ("N22", "N23", "E19", 450.0, 50.0, 1400.0, False),
            ("N23", "N24", "E20", 450.0, 60.0, 1700.0, False),

            # Vertical connections Column 1
            ("N1", "N7", "E21", 400.0, 50.0, 1300.0, False),
            ("N7", "N13", "E22", 400.0, 50.0, 1300.0, False),
            ("N13", "N19", "E23", 400.0, 60.0, 1500.0, False),

            # Column 2
            ("N2", "N8", "E24", 400.0, 45.0, 1400.0, False),
            ("N8", "N14", "E25", 400.0, 45.0, 1400.0, False),
            ("N14", "N20", "E26", 400.0, 50.0, 1500.0, False),

            # Column 3
            ("N3", "N9", "E27", 400.0, 40.0, 1500.0, True),
            ("N9", "N15", "E28", 400.0, 40.0, 1500.0, True),
            ("N15", "N21", "E29", 400.0, 45.0, 1400.0, False),

            # Column 4
            ("N4", "N10", "E30", 400.0, 45.0, 1600.0, False),
            ("N10", "N16", "E31", 400.0, 40.0, 1400.0, False),
            ("N16", "N22", "E32", 400.0, 45.0, 1300.0, False),

            # Column 5
            ("N5", "N11", "E33", 400.0, 50.0, 1500.0, False),
            ("N11", "N17", "E34", 400.0, 50.0, 1500.0, False),
            ("N17", "N23", "E35", 400.0, 50.0, 1400.0, False),

            # Column 6
            ("N6", "N12", "E36", 400.0, 60.0, 1700.0, False),
            ("N12", "N18", "E37", 400.0, 55.0, 1600.0, False),
            ("N18", "N24", "E38", 400.0, 60.0, 1700.0, False),
        ]

        # Add both forward and reverse directions for realism
        for u, v, eid, length, free_speed, cap, is_transit in edges_spec:
            self._add_road_edge(u, v, eid, length, free_speed, cap, is_transit)
            # Add reverse edge
            rev_eid = f"{eid}_R"
            self._add_road_edge(v, u, rev_eid, length, free_speed, cap, is_transit)

    def _add_road_edge(self, u: str, v: str, eid: str, length: float, free_speed: float, capacity: float, is_transit: bool):
        # Calculate free flow travel time in seconds
        free_flow_time = (length / 1000.0) / (free_speed / 3600.0)
        edge_dict = {
            "edge_id": eid,
            "from_node": u,
            "to_node": v,
            "length": length,
            "free_flow_speed": free_speed,
            "current_speed": free_speed,
            "capacity": capacity,
            "flow": 300.0,  # initial base flow
            "travel_time": free_flow_time,
            "congestion": 0.2,
            "is_transit_lane": is_transit,
            "incident_penalty": 0.0,
            "is_blocked": False,
        }
        self.edges_data[eid] = edge_dict
        self.edge_by_nodes[(u, v)] = eid
        self.graph.add_edge(u, v, **edge_dict)

    def compute_bpr_travel_time(self, edge_id: str, flow_override: Optional[float] = None) -> float:
        """
        BPR Volume-Delay Function:
        T = T_0 * (1 + alpha * (Flow / Capacity) ^ beta) + incident_penalty
        """
        edge = self.edges_data[edge_id]
        if edge["is_blocked"]:
            return 99999.0  # blocked / infinite cost
            
        t0 = (edge["length"] / 1000.0) / (edge["free_flow_speed"] / 3600.0)
        flow = flow_override if flow_override is not None else edge["flow"]
        cap = max(100.0, edge["capacity"])
        vc_ratio = flow / cap
        
        # Standard BPR: alpha=0.15, beta=4.0
        bpr_factor = 1.0 + 0.15 * math.pow(vc_ratio, 4.0)
        incident_mult = 1.0 + edge["incident_penalty"]
        
        travel_time = t0 * bpr_factor * incident_mult
        return travel_time

    def update_edge_traffic(self, edge_id: str, vehicle_count_on_edge: int, dt: float = 1.0):
        """
        Updates dynamic speed, flow, congestion, and travel time from current vehicle density.
        """
        edge = self.edges_data[edge_id]
        if edge["is_blocked"]:
            edge["current_speed"] = 2.0
            edge["congestion"] = 1.0
            edge["travel_time"] = 99999.0
            return

        # Estimate equivalent hourly flow rate
        edge["flow"] = max(50.0, vehicle_count_on_edge * 120.0)
        tt = self.compute_bpr_travel_time(edge_id)
        edge["travel_time"] = tt
        
        # Effective speed (km/h)
        speed = (edge["length"] / 1000.0) / (tt / 3600.0)
        edge["current_speed"] = max(5.0, min(edge["free_flow_speed"], speed))
        
        # Congestion metric (0 to 1)
        cong = 1.0 - (edge["current_speed"] / edge["free_flow_speed"])
        edge["congestion"] = max(0.0, min(1.0, cong))

    def get_edge_state(self, edge_id: str) -> EdgeState:
        e = self.edges_data[edge_id]
        return EdgeState(
            edge_id=e["edge_id"],
            from_node=e["from_node"],
            to_node=e["to_node"],
            length=e["length"],
            speed=round(e["current_speed"], 1),
            flow=round(e["flow"], 1),
            travel_time=round(e["travel_time"], 1),
            congestion=round(e["congestion"], 2),
            free_flow_speed=e["free_flow_speed"],
            capacity=e["capacity"]
        )

    def get_all_edge_states(self) -> Dict[str, EdgeState]:
        return {eid: self.get_edge_state(eid) for eid in self.edges_data}

    def find_shortest_path_edges(self, origin: str, destination: str, weight: str = "travel_time") -> List[str]:
        """Returns ordered list of edge IDs connecting origin to destination."""
        try:
            node_path = nx.shortest_path(self.graph, source=origin, target=destination, weight=weight)
            edge_path = []
            for i in range(len(node_path) - 1):
                u = node_path[i]
                v = node_path[i + 1]
                eid = self.edge_by_nodes.get((u, v))
                if eid:
                    edge_path.append(eid)
            return edge_path
        except nx.NetworkXNoPath:
            return []
