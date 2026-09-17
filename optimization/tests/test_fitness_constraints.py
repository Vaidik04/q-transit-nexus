"""
test_fitness_constraints.py — Unit tests for ConstraintManager and MultiObjectiveFitness.
"""

import numpy as np
import pytest

from optimization.config import OptimizationConfig, OptimizationMode
from optimization.constraints import ConstraintManager
from optimization.fitness import MultiObjectiveFitness
from optimization.graph_interface import TransportationGraph
from optimization.route_decoder import DecodedVehicleRoute, RouteDecoder
from optimization.route_encoder import RouteEncoder, VehicleRoutingRequest


def test_capacity_and_incident_constraints():
    tg = TransportationGraph.create_prototype_network()
    config = OptimizationConfig()
    cm = ConstraintManager(tg, config)

    # Set severe accident on E_11_21
    tg.update_edge_state("E_11_21", incident_severity=0.95)

    v1 = VehicleRoutingRequest(vehicle_id="PV1", origin="J_01", destination="J_31", vehicle_type="passenger")
    v2 = VehicleRoutingRequest(vehicle_id="MED1", origin="J_01", destination="J_31", vehicle_type="ambulance", priority="EMERGENCY")
    encoder = RouteEncoder(tg, [v1, v2], k_paths=2)

    # Passenger vehicle routed through incident
    routes_bad = [
        DecodedVehicleRoute("PV1", "passenger", "NORMAL", ["E_01_11", "E_11_21", "E_21_31"], 1200.0, 100.0),
        DecodedVehicleRoute("MED1", "ambulance", "EMERGENCY", ["E_01_11", "E_11_21", "E_21_31"], 1200.0, 50.0),
    ]

    res = cm.evaluate_constraints(routes_bad, encoder)
    assert res.violation_count >= 1
    # Only passenger vehicle receives incident violation; ambulance is allowed priority access
    assert any(v.constraint_type == "INCIDENT" and v.vehicle_id == "PV1" for v in res.violations)
    assert not any(v.constraint_type == "INCIDENT" and v.vehicle_id == "MED1" for v in res.violations)


def test_ev_battery_constraint():
    tg = TransportationGraph.create_prototype_network()
    config = OptimizationConfig(min_ev_soc=20.0)
    cm = ConstraintManager(tg, config)

    # EV with low battery (10% SoC)
    v_ev = VehicleRoutingRequest(vehicle_id="EV1", origin="J_00", destination="J_30", vehicle_type="ev", battery_soc=10.0)
    encoder = RouteEncoder(tg, [v_ev], k_paths=1)

    routes = [
        DecodedVehicleRoute("EV1", "ev", "NORMAL", ["E_00_10", "E_10_20", "E_20_30"], 1200.0, 90.0)
    ]

    res = cm.evaluate_constraints(routes, encoder)
    assert any(v.constraint_type == "EV_BATTERY" for v in res.violations)


def test_multiobjective_fitness_modes():
    tg = TransportationGraph.create_prototype_network()
    v1 = VehicleRoutingRequest(vehicle_id="PV1", origin="J_00", destination="J_30", vehicle_type="passenger")
    v2 = VehicleRoutingRequest(vehicle_id="BUS1", origin="J_01", destination="J_31", vehicle_type="bus", passenger_load=40)
    encoder = RouteEncoder(tg, [v1, v2], k_paths=2)

    routes = [
        DecodedVehicleRoute("PV1", "passenger", "NORMAL", ["E_00_10", "E_10_20", "E_20_30"], 1200.0, 80.0),
        DecodedVehicleRoute("BUS1", "bus", "HIGH", ["E_01_11", "E_11_21", "E_21_31"], 1200.0, 90.0),
    ]

    for mode in [
        OptimizationMode.BALANCED,
        OptimizationMode.FAST,
        OptimizationMode.EMERGENCY,
        OptimizationMode.GREEN,
        OptimizationMode.ECO,
        OptimizationMode.PUBLIC_TRANSPORT_PRIORITY,
        OptimizationMode.TRANSIT,
    ]:
        cfg = OptimizationConfig(mode=mode)
        fitness_engine = MultiObjectiveFitness(tg, cfg)
        breakdown = fitness_engine.evaluate(routes, encoder)
        assert breakdown.total_fitness > 0.0
        assert breakdown.travel_time_sec > 0.0


def test_extended_constraints_all():
    tg = TransportationGraph.create_prototype_network()
    cfg = OptimizationConfig()
    cm = ConstraintManager(tg, cfg)

    # 1. Vehicle Capacity violation
    v_cap = VehicleRoutingRequest(vehicle_id="V_CAP", origin="J_00", destination="J_10", capacity=2.0, current_load=5.0)
    # 2. Route duration violation
    v_dur = VehicleRoutingRequest(vehicle_id="V_DUR", origin="J_00", destination="J_30", max_duration_sec=30.0)
    # 3. Time window deadline violation
    v_tw = VehicleRoutingRequest(vehicle_id="V_TW", origin="J_00", destination="J_30", departure_time_sec=0.0, time_window_latest_sec=40.0)
    # 4. Maximum distance violation
    v_dist = VehicleRoutingRequest(vehicle_id="V_DIST", origin="J_00", destination="J_30", max_distance_m=500.0)
    # 5. Vehicle availability violation
    v_unavail = VehicleRoutingRequest(vehicle_id="V_UNAV", origin="J_00", destination="J_10", is_available=False)
    # 6. Depot return violation
    v_depot = VehicleRoutingRequest(vehicle_id="V_DEP", origin="J_00", destination="J_10", depot_node="J_32")

    vehicles = [v_cap, v_dur, v_tw, v_dist, v_unavail, v_depot]
    encoder = RouteEncoder(tg, vehicles, k_paths=1)

    routes = [
        DecodedVehicleRoute("V_CAP", "passenger", "NORMAL", ["E_00_10"], 400.0, 30.0),
        DecodedVehicleRoute("V_DUR", "passenger", "NORMAL", ["E_00_10", "E_10_20", "E_20_30"], 1200.0, 90.0),
        DecodedVehicleRoute("V_TW", "passenger", "NORMAL", ["E_00_10", "E_10_20", "E_20_30"], 1200.0, 90.0),
        DecodedVehicleRoute("V_DIST", "passenger", "NORMAL", ["E_00_10", "E_10_20", "E_20_30"], 1200.0, 90.0),
        DecodedVehicleRoute("V_UNAV", "passenger", "NORMAL", ["E_00_10"], 400.0, 30.0),
        DecodedVehicleRoute("V_DEP", "passenger", "NORMAL", ["E_00_10"], 400.0, 30.0),
    ]

    res = cm.evaluate_constraints(routes, encoder)
    v_types = {v.constraint_type for v in res.violations}

    assert "VEHICLE_CAPACITY" in v_types
    assert "ROUTE_DURATION" in v_types
    assert "TIME_WINDOW" in v_types
    assert "MAX_DISTANCE" in v_types
    assert "VEHICLE_AVAILABILITY" in v_types
    assert "DEPOT_REQUIREMENT" in v_types
    assert res.total_penalty > 0.0


def test_prediction_aware_fitness_sensitivity():
    """Verify that changing predicted travel time directly alters prediction-aware fitness."""
    tg = TransportationGraph.create_prototype_network()
    edge = tg.get_edge("E_01_11")
    assert edge is not None

    # Base current travel time at 50 km/h: 400m / (50 * 1000 / 3600) = 28.8s
    # Set predicted T+5 travel time higher (e.g. 150.0s)
    edge.predicted_t5_sec = 150.0

    v = VehicleRoutingRequest(vehicle_id="V_PRED", origin="J_01", destination="J_11")
    encoder = RouteEncoder(tg, [v], k_paths=1)
    decoder = RouteDecoder(encoder, tg)

    # Decode at horizon 0 (current conditions)
    routes_h0 = decoder.decode_particle(np.array([0.0]), future_horizon_min=0)
    cfg = OptimizationConfig()
    mf = MultiObjectiveFitness(tg, cfg)
    fit_h0 = mf.evaluate(routes_h0, encoder, future_horizon_min=0)

    # Decode at horizon 5 (ML predicted conditions)
    routes_h5 = decoder.decode_particle(np.array([0.0]), future_horizon_min=5)
    fit_h5 = mf.evaluate(routes_h5, encoder, future_horizon_min=5)

    # Predicted cost at T+5 must be strictly greater than at T=0
    assert fit_h5.travel_time_sec > fit_h0.travel_time_sec
    assert fit_h5.total_fitness > fit_h0.total_fitness

    # Increasing predicted T+5 travel time further must increase prediction-aware fitness monotonically
    edge.predicted_t5_sec = 300.0
    fit_h5_higher = mf.evaluate(routes_h5, encoder, future_horizon_min=5)
    assert fit_h5_higher.travel_time_sec > fit_h5.travel_time_sec
    assert fit_h5_higher.total_fitness > fit_h5.total_fitness


def test_bpr_congestion_penalty():
    """Verify that BPR volume-to-capacity latency function penalizes vehicle crowding on a single edge."""
    tg = TransportationGraph.create_prototype_network()
    cfg = OptimizationConfig(enable_bpr_latency=True, bpr_alpha=0.15, bpr_beta=2.0)
    mf = MultiObjectiveFitness(tg, cfg)

    # Scenario A: 10 vehicles all crowded onto the SAME edge E_00_10
    vehicles_crowded = [
        VehicleRoutingRequest(vehicle_id=f"V_CROWD_{i}", origin="J_00", destination="J_10")
        for i in range(10)
    ]
    encoder_crowded = RouteEncoder(tg, vehicles_crowded, k_paths=1)
    routes_crowded = [
        DecodedVehicleRoute(v.vehicle_id, "passenger", "NORMAL", ["E_00_10"], 400.0, 30.0)
        for v in vehicles_crowded
    ]
    fit_crowded = mf.evaluate(routes_crowded, encoder_crowded, future_horizon_min=0)

    # Scenario B: 10 vehicles dispersed across 5 different edges (2 vehicles per edge)
    edges_pool = ["E_00_10", "E_01_11", "E_02_12", "E_10_20", "E_11_21"]
    vehicles_dispersed = [
        VehicleRoutingRequest(vehicle_id=f"V_DISP_{i}", origin="J_00", destination="J_10")
        for i in range(10)
    ]
    encoder_dispersed = RouteEncoder(tg, vehicles_dispersed, k_paths=1)
    routes_dispersed = [
        DecodedVehicleRoute(v.vehicle_id, "passenger", "NORMAL", [edges_pool[i % len(edges_pool)]], 400.0, 30.0)
        for i, v in enumerate(vehicles_dispersed)
    ]
    fit_dispersed = mf.evaluate(routes_dispersed, encoder_dispersed, future_horizon_min=0)

    # Crowded assignment must suffer higher travel time and higher fitness penalty due to BPR non-linear latency
    assert fit_crowded.travel_time_sec > fit_dispersed.travel_time_sec
    assert fit_crowded.total_fitness > fit_dispersed.total_fitness
