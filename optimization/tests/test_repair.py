"""
test_repair.py — Unit tests for RouteRepairer.
"""

from optimization.graph_interface import TransportationGraph
from optimization.repair import RouteRepairer


def test_valid_route_check():
    tg = TransportationGraph.create_prototype_network()
    repairer = RouteRepairer(tg)

    valid_route = ["E_00_10", "E_10_20", "E_20_30"]
    assert repairer.is_valid_route(valid_route) is True

    # Disconnected route
    invalid_route = ["E_00_10", "E_21_31"]
    assert repairer.is_valid_route(invalid_route) is False


def test_cycle_removal():
    tg = TransportationGraph.create_prototype_network()
    repairer = RouteRepairer(tg)

    # Route that goes E_00_10 -> E_10_11 -> E_11_10 (cycle!) -> E_10_20
    looped_route = ["E_00_10", "E_10_11", "E_11_10", "E_10_20"]
    cleaned = repairer.remove_cycles(looped_route)
    assert repairer.is_valid_route(cleaned) is True
    # Should have pruned the intermediate loop
    assert cleaned == ["E_00_10", "E_10_20"]


def test_repair_disconnection():
    tg = TransportationGraph.create_prototype_network()
    repairer = RouteRepairer(tg)

    # Disconnected edges from J_00 to J_30
    gap_route = ["E_00_10", "E_20_30"]
    repaired = repairer.repair_disconnections(gap_route, origin_node="J_00", destination_node="J_30")
    assert repairer.is_valid_route(repaired) is True
    assert repaired[0] == "E_00_10"
    assert repaired[-1] == "E_20_30"
    assert "E_10_20" in repaired
