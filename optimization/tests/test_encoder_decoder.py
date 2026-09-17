"""
test_encoder_decoder.py — Unit tests for RouteEncoder and RouteDecoder.
"""

import numpy as np
import pytest

from optimization.graph_interface import TransportationGraph
from optimization.route_decoder import RouteDecoder
from optimization.route_encoder import RouteEncoder, VehicleRoutingRequest


def test_encoder_initialization_and_bounds():
    tg = TransportationGraph.create_prototype_network()
    vehicles = [
        VehicleRoutingRequest(vehicle_id="V1", origin="J_00", destination="J_30"),
        VehicleRoutingRequest(vehicle_id="V2", origin="J_01", destination="J_31"),
    ]
    encoder = RouteEncoder(tg, vehicles, k_paths=4)
    assert encoder.dimension == 2
    assert len(encoder.candidate_paths["V1"]) >= 1

    pop = encoder.initialize_population(n_particles=10, seed=42)
    assert pop.shape == (10, 2)
    assert np.all(pop >= 0.0)
    assert np.all(pop <= 1.0)


def test_decoder_mapping_and_api():
    tg = TransportationGraph.create_prototype_network()
    vehicles = [
        VehicleRoutingRequest(vehicle_id="V1", origin="J_00", destination="J_30"),
        VehicleRoutingRequest(vehicle_id="V2", origin="J_01", destination="J_31"),
    ]
    encoder = RouteEncoder(tg, vehicles, k_paths=3)
    decoder = RouteDecoder(encoder, tg)

    # Particle position [0.1, 0.9]
    particle = np.array([0.1, 0.9])
    decoded = decoder.decode_particle(particle)
    assert len(decoded) == 2
    assert decoded[0].vehicle_id == "V1"
    assert decoded[1].vehicle_id == "V2"
    assert len(decoded[0].route) > 0

    api_decisions = decoder.to_api_decisions(decoded)
    assert len(api_decisions) == 2
    assert "vehicle_id" in api_decisions[0]
    assert "route" in api_decisions[0]
