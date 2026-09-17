"""
test_graph_interface.py — Unit tests for TransportationGraph.
"""

import pytest

from optimization.graph_interface import TransportationGraph


def test_add_node_and_edges():
    tg = TransportationGraph()
    tg.add_node("A", x=0.0, y=0.0)
    tg.add_node("B", x=100.0, y=0.0)
    e = tg.add_edge("E1", "A", "B", length_m=500.0, freeflow_speed_kmh=50.0)

    assert e.edge_id == "E1"
    assert tg.get_edge("E1") is not None
    assert tg.get_edge_between("A", "B") is not None
    # 500m at 50 km/h = 500 / (50 * 1000 / 3600) = 36.0 seconds
    assert pytest.approx(e.travel_time_sec, rel=1e-2) == 36.0


def test_dynamic_traffic_update():
    tg = TransportationGraph()
    tg.add_edge("E1", "A", "B", length_m=1000.0, freeflow_speed_kmh=60.0)
    
    # Update speed to 30 km/h
    tg.update_edge_state("E1", speed_kmh=30.0, flow_vph=500.0, occupancy=0.5)
    e = tg.get_edge("E1")
    assert e.current_speed_kmh == 30.0
    assert e.congestion_score == 0.5
    # 1000m at 30 km/h = 1000 / (30 * 1000 / 3600) = 120.0 seconds
    assert pytest.approx(e.travel_time_sec, rel=1e-2) == 120.0


def test_ml_prediction_ingestion():
    tg = TransportationGraph()
    tg.add_edge("E14", "J1", "J2", length_m=800.0)

    # Ingest ML predictions
    preds = [
        {"edge_id": "E14", "t5": 45.2, "t10": 55.8, "t15": 72.1, "congestion": 0.42}
    ]
    tg.ingest_ml_predictions(preds)
    e = tg.get_edge("E14")
    assert e.predicted_t5_sec == 45.2
    assert e.predicted_t10_sec == 55.8
    assert e.predicted_t15_sec == 72.1
    assert e.get_travel_time_for_horizon(5) == 45.2
    assert e.get_travel_time_for_horizon(10) == 55.8


def test_prototype_grid_creation():
    tg = TransportationGraph.create_prototype_network()
    assert len(tg.graph.nodes) == 12
    assert len(tg.edges_by_id) >= 34
    assert "J_00" in tg.graph.nodes
    assert "J_32" in tg.graph.nodes
    assert "E_11_21" in tg.edges_by_id


def test_k_shortest_paths():
    tg = TransportationGraph.create_prototype_network()
    paths = tg.find_k_shortest_edge_paths("J_00", "J_32", k=5)
    assert len(paths) >= 2
    for p in paths:
        assert len(p) >= 5 # Needs at least 5 hops across the grid
