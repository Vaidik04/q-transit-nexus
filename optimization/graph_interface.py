"""
graph_interface.py — NetworkX-based Transportation Graph interface.

Wraps a directed graph G = (V, E) where:
- V: Intersections / junctions (coordinates, signal status)
- E: Directed road edges (length, freeflow speed, dynamic speed, capacity, occupancy,
     ML predicted future costs for T+5, T+10, T+15, and congestion).

Provides:
- Dynamic travel time and impedance calculations
- Ingestion of ML EdgePrediction objects from 137ML/prediction/inference.py
- K-shortest loopless candidate path generation (Yen's algorithm via NetworkX)
- Standard prototype network generator (4x3 grid with 34 corridor edges) matching SIH spec.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class EdgeData:
    """Attributes of a directed road edge in the transportation network."""
    edge_id: str
    u: str                           # Origin junction node ID
    v: str                           # Destination junction node ID
    length_m: float                  # Length in meters
    speed_limit_kmh: float = 50.0    # Speed limit in km/h
    freeflow_speed_kmh: float = 50.0 # Free-flow speed in km/h
    capacity_vph: float = 1200.0     # Capacity (vehicles per hour)
    num_lanes: int = 2               # Number of lanes
    is_oneway: bool = True           # Directed edge
    
    # Dynamic runtime state
    current_speed_kmh: float = 50.0
    occupancy: float = 0.0           # [0, 1]
    flow_vph: float = 0.0            # Vehicle count / flow rate
    incident_severity: float = 0.0   # [0, 1] (0 = clear, 1 = complete blockage)
    
    # ML predicted travel times (seconds)
    predicted_t5_sec: Optional[float] = None
    predicted_t10_sec: Optional[float] = None
    predicted_t15_sec: Optional[float] = None
    congestion_score: float = 0.0    # [0, 1]

    @property
    def travel_time_sec(self) -> float:
        """Calculates current travel time based on dynamic speed and incident severity."""
        effective_speed = max(self.current_speed_kmh * (1.0 - 0.9 * self.incident_severity), 1.0)
        speed_mps = effective_speed * (1000.0 / 3600.0)
        return self.length_m / speed_mps

    def get_travel_time_for_horizon(self, horizon_min: int = 0) -> float:
        """Returns predicted travel time for future horizon if available, else current."""
        if horizon_min == 5 and self.predicted_t5_sec is not None:
            return self.predicted_t5_sec
        elif horizon_min == 10 and self.predicted_t10_sec is not None:
            return self.predicted_t10_sec
        elif horizon_min == 15 and self.predicted_t15_sec is not None:
            return self.predicted_t15_sec
        return self.travel_time_sec


@dataclass
class NodeData:
    """Attributes of an intersection/junction node."""
    node_id: str
    x: float = 0.0
    y: float = 0.0
    is_signalized: bool = False
    signal_cycle_sec: float = 60.0
    current_phase: int = 0


class TransportationGraph:
    """Directed multigraph / DiGraph representation of the urban street network."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.edges_by_id: Dict[str, EdgeData] = {}
        self.node_metadata: Dict[str, NodeData] = {}
        self._path_cache: Dict[Tuple[str, str, int], List[List[str]]] = {}

    def add_node(
        self,
        node_id: str,
        x: float = 0.0,
        y: float = 0.0,
        is_signalized: bool = False,
        signal_cycle_sec: float = 60.0,
    ) -> None:
        """Adds a junction node to the graph."""
        self.graph.add_node(node_id, x=x, y=y, is_signalized=is_signalized)
        self.node_metadata[node_id] = NodeData(
            node_id=node_id,
            x=x,
            y=y,
            is_signalized=is_signalized,
            signal_cycle_sec=signal_cycle_sec,
        )

    def add_edge(
        self,
        edge_id: str,
        u: str,
        v: str,
        length_m: float,
        freeflow_speed_kmh: float = 50.0,
        capacity_vph: float = 1200.0,
        num_lanes: int = 2,
    ) -> EdgeData:
        """Adds a directed road edge between nodes u and v."""
        if not self.graph.has_node(u):
            self.add_node(u)
        if not self.graph.has_node(v):
            self.add_node(v)

        edge_data = EdgeData(
            edge_id=edge_id,
            u=u,
            v=v,
            length_m=length_m,
            freeflow_speed_kmh=freeflow_speed_kmh,
            speed_limit_kmh=freeflow_speed_kmh,
            current_speed_kmh=freeflow_speed_kmh,
            capacity_vph=capacity_vph,
            num_lanes=num_lanes,
        )

        freeflow_mps = freeflow_speed_kmh * (1000.0 / 3600.0)
        freeflow_time = length_m / max(freeflow_mps, 1.0)

        self.edges_by_id[edge_id] = edge_data
        self.graph.add_edge(
            u,
            v,
            edge_id=edge_id,
            length=length_m,
            freeflow_time=freeflow_time,
            weight=edge_data.travel_time_sec,
            data=edge_data,
        )
        self.invalidate_path_cache()
        return edge_data

    def invalidate_path_cache(self) -> None:
        """Clears precomputed path caches when topology or weights are updated."""
        self._path_cache.clear()

    def update_edge_state(
        self,
        edge_id: str,
        speed_kmh: Optional[float] = None,
        flow_vph: Optional[float] = None,
        occupancy: Optional[float] = None,
        incident_severity: Optional[float] = None,
    ) -> None:
        """Updates live edge conditions from simulation or sensor feed."""
        if edge_id not in self.edges_by_id:
            logger.warning("Edge %s not found in graph.", edge_id)
            return

        edge = self.edges_by_id[edge_id]
        if speed_kmh is not None:
            edge.current_speed_kmh = max(speed_kmh, 1.0)
            edge.congestion_score = max(0.0, min(1.0, 1.0 - (edge.current_speed_kmh / edge.freeflow_speed_kmh)))
        if flow_vph is not None:
            edge.flow_vph = max(flow_vph, 0.0)
        if occupancy is not None:
            edge.occupancy = max(0.0, min(1.0, occupancy))
        if incident_severity is not None:
            edge.incident_severity = max(0.0, min(1.0, incident_severity))

        # Update edge weight in networkx graph
        self.graph[edge.u][edge.v]["weight"] = edge.travel_time_sec

    def ingest_ml_predictions(self, predictions: Sequence[Any]) -> None:
        """Ingests EdgePrediction objects or dicts from the ML prediction module."""
        for pred in predictions:
            if hasattr(pred, "edge_id"):
                eid = pred.edge_id
                t5 = getattr(pred, "t5", None)
                t10 = getattr(pred, "t10", None)
                t15 = getattr(pred, "t15", None)
                cong = getattr(pred, "congestion", 0.0)
            elif isinstance(pred, dict):
                eid = pred.get("edge_id")
                t5 = pred.get("t5")
                t10 = pred.get("t10")
                t15 = pred.get("t15")
                cong = pred.get("congestion", 0.0)
            else:
                continue

            if eid in self.edges_by_id:
                edge = self.edges_by_id[eid]
                edge.predicted_t5_sec = t5
                edge.predicted_t10_sec = t10
                edge.predicted_t15_sec = t15
                if cong is not None:
                    edge.congestion_score = float(cong)

    def get_edge(self, edge_id: str) -> Optional[EdgeData]:
        return self.edges_by_id.get(edge_id)

    def get_edge_between(self, u: str, v: str) -> Optional[EdgeData]:
        if self.graph.has_edge(u, v):
            return self.graph[u][v].get("data")
        return None

    def find_k_shortest_node_paths(
        self,
        source_node: str,
        target_node: str,
        k: int = 5,
        weight_attr: str = "weight",
    ) -> List[List[str]]:
        """Finds up to k loopless shortest paths between two nodes using Yen's algorithm."""
        cache_key = (source_node, target_node, k)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        if not self.graph.has_node(source_node) or not self.graph.has_node(target_node):
            return []

        try:
            paths = list(
                nx.shortest_simple_paths(
                    self.graph,
                    source=source_node,
                    target=target_node,
                    weight=weight_attr,
                )
            )
            result = paths[:k]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            result = []

        self._path_cache[cache_key] = result
        return result

    def node_path_to_edge_path(self, node_path: Sequence[str]) -> List[str]:
        """Converts a sequence of junction nodes [u1, u2, ..., un] into edge IDs [e1, e2, ...]."""
        if len(node_path) < 2:
            return []
        edge_ids: List[str] = []
        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            edge_data = self.get_edge_between(u, v)
            if edge_data:
                edge_ids.append(edge_data.edge_id)
            else:
                return []  # Infeasible path
        return edge_ids

    def find_k_shortest_edge_paths(
        self,
        source_node_or_edge: str,
        target_node_or_edge: str,
        k: int = 5,
        weight_attr: str = "weight",
    ) -> List[List[str]]:
        """Returns candidate paths as sequences of edge IDs.
        Accepts either node IDs or edge IDs as origin and destination.
        """
        # Resolve source node
        prefix_edge: Optional[str] = None
        if source_node_or_edge in self.edges_by_id:
            src_edge = self.edges_by_id[source_node_or_edge]
            source_node = src_edge.v
            prefix_edge = src_edge.edge_id
        else:
            source_node = source_node_or_edge

        # Resolve target node
        suffix_edge: Optional[str] = None
        if target_node_or_edge in self.edges_by_id:
            tgt_edge = self.edges_by_id[target_node_or_edge]
            target_node = tgt_edge.u
            suffix_edge = tgt_edge.edge_id
        else:
            target_node = target_node_or_edge

        if source_node == target_node:
            res: List[str] = []
            if prefix_edge:
                res.append(prefix_edge)
            if suffix_edge and suffix_edge != prefix_edge:
                res.append(suffix_edge)
            return [res] if res else []

        node_paths = self.find_k_shortest_node_paths(source_node, target_node, k=k, weight_attr=weight_attr)
        candidate_edge_paths: List[List[str]] = []
        for np in node_paths:
            ep = self.node_path_to_edge_path(np)
            full_path: List[str] = []
            if prefix_edge:
                full_path.append(prefix_edge)
            full_path.extend(ep)
            if suffix_edge and (not full_path or full_path[-1] != suffix_edge):
                full_path.append(suffix_edge)
            if full_path and full_path not in candidate_edge_paths:
                candidate_edge_paths.append(full_path)

        return candidate_edge_paths

    @classmethod
    def create_prototype_network(cls) -> "TransportationGraph":
        """Creates the canonical 4x3 grid network (12 intersections, 34 edges)
        corresponding to the SIH prototype corridor specification in README.md.
        """
        tg = cls()

        # Create 12 intersections: J_00 to J_32 (x: 0..3, y: 0..2)
        spacing_m = 400.0  # 400m block lengths
        for x in range(4):
            for y in range(3):
                node_id = f"J_{x}{y}"
                is_signal = (x in (1, 2) and y == 1) # J_11 and J_21 are key signalized intersections
                tg.add_node(
                    node_id=node_id,
                    x=x * spacing_m,
                    y=y * spacing_m,
                    is_signalized=is_signal,
                    signal_cycle_sec=60.0 if is_signal else 0.0,
                )

        # Horizontal bidirectional edges
        for y in range(3):
            for x in range(3):
                u = f"J_{x}{y}"
                v = f"J_{x+1}{y}"
                # Eastbound
                tg.add_edge(
                    edge_id=f"E_{x}{y}_{x+1}{y}",
                    u=u,
                    v=v,
                    length_m=spacing_m,
                    freeflow_speed_kmh=50.0 if y != 1 else 60.0, # Main artery on y=1
                    capacity_vph=1500.0 if y == 1 else 1000.0,
                    num_lanes=3 if y == 1 else 2,
                )
                # Westbound
                tg.add_edge(
                    edge_id=f"E_{x+1}{y}_{x}{y}",
                    u=v,
                    v=u,
                    length_m=spacing_m,
                    freeflow_speed_kmh=50.0 if y != 1 else 60.0,
                    capacity_vph=1500.0 if y == 1 else 1000.0,
                    num_lanes=3 if y == 1 else 2,
                )

        # Vertical bidirectional edges
        for x in range(4):
            for y in range(2):
                u = f"J_{x}{y}"
                v = f"J_{x}{y+1}"
                # Northbound
                tg.add_edge(
                    edge_id=f"E_{x}{y}_{x}{y+1}",
                    u=u,
                    v=v,
                    length_m=spacing_m,
                    freeflow_speed_kmh=45.0,
                    capacity_vph=1000.0,
                    num_lanes=2,
                )
                # Southbound
                tg.add_edge(
                    edge_id=f"E_{x}{y+1}_{x}{y}",
                    u=v,
                    v=u,
                    length_m=spacing_m,
                    freeflow_speed_kmh=45.0,
                    capacity_vph=1000.0,
                    num_lanes=2,
                )

        # Ensure corridor alias E14 maps to central bottleneck edge E_11_21
        if "E_11_21" in tg.edges_by_id:
            tg.edges_by_id["E14"] = tg.edges_by_id["E_11_21"]

        return tg
