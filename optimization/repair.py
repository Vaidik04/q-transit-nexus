"""
repair.py — Graph connectivity and route validation repair operator.

Ensures generated and mutated vehicle routes are:
- Continuous (end node of edge i equals start node of edge i+1)
- Loop-free (eliminates wasteful intermediate cycles)
- Connected from vehicle origin to destination
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import networkx as nx

from optimization.graph_interface import TransportationGraph

logger = logging.getLogger(__name__)


class RouteRepairer:
    """Repairs infeasible or disconnected route edge sequences into valid paths."""

    def __init__(self, graph: TransportationGraph) -> None:
        self.graph = graph

    def is_valid_route(self, route: Sequence[str]) -> bool:
        """Verifies if an edge sequence forms a valid, contiguous directed path."""
        if not route:
            return False
        for i in range(len(route) - 1):
            e1 = self.graph.get_edge(route[i])
            e2 = self.graph.get_edge(route[i + 1])
            if not e1 or not e2:
                return False
            if e1.v != e2.u:
                return False
        return True

    def remove_cycles(self, route: Sequence[str]) -> List[str]:
        """Eliminates intermediate loops/cycles from an edge sequence."""
        if len(route) <= 1:
            return list(route)

        # Convert to node sequence: [u0, v0, v1, ..., vn]
        nodes: List[str] = []
        for i, edge_id in enumerate(route):
            edge = self.graph.get_edge(edge_id)
            if not edge:
                return list(route)
            if i == 0:
                nodes.append(edge.u)
            nodes.append(edge.v)

        # Detect repeated nodes and prune cycles
        visited_nodes: Dict[str, int] = {}
        cleaned_nodes: List[str] = []
        i = 0
        while i < len(nodes):
            curr_node = nodes[i]
            if curr_node in visited_nodes:
                # Cycle detected! Truncate back to first occurrence
                prev_idx = visited_nodes[curr_node]
                # Remove intermediate node entries from dictionary
                for node in cleaned_nodes[prev_idx + 1:]:
                    if node in visited_nodes and visited_nodes[node] > prev_idx:
                        del visited_nodes[node]
                cleaned_nodes = cleaned_nodes[:prev_idx + 1]
            else:
                visited_nodes[curr_node] = len(cleaned_nodes)
                cleaned_nodes.append(curr_node)
            i += 1

        # Convert cleaned nodes back to edge sequence
        cleaned_edges = self.graph.node_path_to_edge_path(cleaned_nodes)
        return cleaned_edges if cleaned_edges else list(route)

    def repair_disconnections(
        self,
        route: Sequence[str],
        origin_node: str,
        destination_node: str,
    ) -> List[str]:
        """Repairs disjoint gaps in an edge sequence by splicing shortest sub-paths."""
        if not route:
            return self._fallback_shortest_path(origin_node, destination_node)

        repaired_nodes: List[str] = []
        first_edge = self.graph.get_edge(route[0])
        if not first_edge:
            return self._fallback_shortest_path(origin_node, destination_node)

        # Ensure start at origin_node
        if first_edge.u != origin_node:
            bridge = self._get_node_subpath(origin_node, first_edge.u)
            repaired_nodes.extend(bridge[:-1])

        for i in range(len(route)):
            curr_edge = self.graph.get_edge(route[i])
            if not curr_edge:
                continue
            if not repaired_nodes:
                repaired_nodes.append(curr_edge.u)
            elif repaired_nodes[-1] != curr_edge.u:
                # Bridge gap between repaired_nodes[-1] and curr_edge.u
                bridge = self._get_node_subpath(repaired_nodes[-1], curr_edge.u)
                repaired_nodes.extend(bridge[1:-1])
            repaired_nodes.append(curr_edge.v)

        # Ensure end at destination_node
        if repaired_nodes and repaired_nodes[-1] != destination_node:
            bridge = self._get_node_subpath(repaired_nodes[-1], destination_node)
            repaired_nodes.extend(bridge[1:])

        cleaned_edges = self.graph.node_path_to_edge_path(repaired_nodes)
        if cleaned_edges:
            return self.remove_cycles(cleaned_edges)
        return self._fallback_shortest_path(origin_node, destination_node)

    def _get_node_subpath(self, u: str, v: str) -> List[str]:
        """Finds shortest node path between u and v."""
        if u == v:
            return [u]
        try:
            return nx.shortest_path(self.graph.graph, source=u, target=v, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [u, v]

    def _fallback_shortest_path(self, u: str, v: str) -> List[str]:
        """Finds fallback shortest edge path between u and v."""
        node_path = self._get_node_subpath(u, v)
        edge_path = self.graph.node_path_to_edge_path(node_path)
        return edge_path if edge_path else []
