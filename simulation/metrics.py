"""
Module: simulation.metrics
Description:
    Collects, aggregates, and benchmarks network-wide performance metrics for Q-Transit Nexus.
    Tracks travel time, average speed, waiting time, queue lengths, vehicle throughput,
    transit bus delay, passenger delay, energy/fuel estimates, and CO₂ emissions.
    Includes automated BEFORE vs AFTER scenario comparison with percentage deltas.

What it reads:
    - TraCI live telemetry (vehicles, lanes, emissions)
    - Digital Twin state snapshots

What it writes:
    - Summary metrics dictionaries, BEFORE/AFTER comparison tables, JSON reports, CSV logs

Dependencies:
    - Depends on: traci_controller.py
    - Depended on by: scenario_manager.py, simulation/scenarios/*.py, demo.py, dashboard API
"""

import json
import csv
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class NetworkMetrics:
    simulation_time: float
    active_vehicles: int
    completed_trips: int
    average_speed_kmh: float
    total_waiting_time_s: float
    average_waiting_time_s: float
    total_travel_time_s: float
    average_travel_time_s: float
    average_queue_length: float
    throughput_veh_per_hr: float
    bus_average_delay_s: float
    passenger_total_delay_s: float
    estimated_fuel_liters: float
    estimated_co2_kg: float
    emergency_response_time_s: Optional[float] = None

class MetricsCollector:
    """
    Accumulates time-series and aggregate statistics across the simulation run,
    and computes formal BEFORE vs AFTER comparison benchmarks.
    """

    def __init__(self, controller):
        self.controller = controller
        self.history: List[NetworkMetrics] = []
        self._ambulance_depart: Optional[float] = None
        self._ambulance_arrived: Optional[float] = None

    def record_ambulance_dispatch(self, t: float):
        self._ambulance_depart = t

    def record_ambulance_arrival(self, t: float):
        self._ambulance_arrived = t

    def capture_snapshot(self) -> NetworkMetrics:
        """Computes current instantaneous and cumulative metrics from TraCI."""
        traci = self.controller.traci
        sim_time = self.controller.get_time()

        if not self.controller.is_connected or not traci:
            dummy = NetworkMetrics(0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)
            self.history.append(dummy)
            return dummy

        try:
            veh_ids = traci.vehicle.getIDList()
            n_active = len(veh_ids)
            n_completed = traci.simulation.getArrivedNumber()

            # Speed
            speeds = [traci.vehicle.getSpeed(v) for v in veh_ids]
            avg_speed_ms = (sum(speeds) / len(speeds)) if speeds else 0.0
            avg_speed_kmh = avg_speed_ms * 3.6

            # Waiting times
            wait_times = [traci.vehicle.getWaitingTime(v) for v in veh_ids]
            tot_wait = sum(wait_times)
            avg_wait = (tot_wait / len(wait_times)) if wait_times else 0.0

            # Travel times
            travel_times = []
            for v in veh_ids:
                try:
                    dep = float(traci.vehicle.getDeparture(v))
                    travel_times.append(max(0.0, sim_time - dep))
                except Exception:
                    pass
            tot_travel = sum(travel_times)
            avg_travel = (tot_travel / len(travel_times)) if travel_times else 0.0

            # Queue length estimation (vehicles stopped with speed < 0.5 m/s)
            queued = sum(1 for s in speeds if s < 0.5)

            # Fuel and CO2 emissions
            co2_mg_s = sum(traci.vehicle.getCO2Emission(v) for v in veh_ids)
            fuel_ml_s = sum(traci.vehicle.getFuelConsumption(v) for v in veh_ids)
            co2_kg = (co2_mg_s / 1e6) * (sim_time / 3600.0)
            fuel_l = (fuel_ml_s / 1000.0) * (sim_time / 3600.0)

            # Bus & Passenger Delays
            bus_delays = []
            passenger_delay_total = 0.0
            for v in veh_ids:
                if "bus" in traci.vehicle.getTypeID(v).lower():
                    try:
                        d = float(traci.vehicle.getParameter(v, "scheduled_delay") or "0.0")
                        load = int(traci.vehicle.getParameter(v, "passenger_load") or "25")
                        bus_delays.append(d)
                        passenger_delay_total += (d * load)
                    except Exception:
                        pass
            avg_bus_delay = (sum(bus_delays) / len(bus_delays)) if bus_delays else 0.0

            # Ambulance response time
            amb_response = None
            if "ambulance_MED01" in veh_ids and self._ambulance_depart is None:
                self._ambulance_depart = sim_time
            if self._ambulance_depart is not None:
                if "ambulance_MED01" not in veh_ids and self._ambulance_arrived is None:
                    self._ambulance_arrived = sim_time
                if self._ambulance_arrived is not None:
                    amb_response = self._ambulance_arrived - self._ambulance_depart
                else:
                    amb_response = sim_time - self._ambulance_depart

            through = (n_completed / max(1.0, sim_time)) * 3600.0

            metric = NetworkMetrics(
                simulation_time=round(sim_time, 1),
                active_vehicles=n_active,
                completed_trips=n_completed,
                average_speed_kmh=round(avg_speed_kmh, 2),
                total_waiting_time_s=round(tot_wait, 1),
                average_waiting_time_s=round(avg_wait, 2),
                total_travel_time_s=round(tot_travel, 1),
                average_travel_time_s=round(avg_travel, 2),
                average_queue_length=float(queued),
                throughput_veh_per_hr=round(through, 1),
                bus_average_delay_s=round(avg_bus_delay, 2),
                passenger_total_delay_s=round(passenger_delay_total, 1),
                estimated_fuel_liters=round(fuel_l, 3),
                estimated_co2_kg=round(co2_kg, 3),
                emergency_response_time_s=round(amb_response, 1) if amb_response is not None else None
            )
            self.history.append(metric)
            return metric

        except Exception as e:
            fallback = NetworkMetrics(sim_time, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None)
            self.history.append(fallback)
            return fallback

    def get_summary(self) -> NetworkMetrics:
        """Returns the most recent snapshot as the overall run summary."""
        return self.history[-1] if self.history else self.capture_snapshot()

    def export_json(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = [asdict(m) for m in self.history]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, filepath: str):
        if not self.history:
            return
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        fields = list(asdict(self.history[0]).keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for m in self.history:
                writer.writerow(asdict(m))

def compare_metrics(before: NetworkMetrics, after: NetworkMetrics,
                    label_before: str = "BEFORE (Incident / Unoptimized)",
                    label_after: str = "AFTER (AT-DQPSO / Preempted)") -> Dict[str, Any]:
    """
    Computes rigorous side-by-side comparison with percentage changes and formatted text table.
    """
    def calc_delta(b_val, a_val, higher_is_better=False):
        if b_val is None or a_val is None or b_val == 0:
            pct = 0.0
        else:
            pct = ((a_val - b_val) / b_val) * 100.0
        # Determine improvement
        if higher_is_better:
            improved = pct > 0
        else:
            improved = pct < 0
        return round(pct, 2), improved

    comparison_items = [
        ("Average Speed (km/h)", before.average_speed_kmh, after.average_speed_kmh, True),
        ("Average Travel Time (s)", before.average_travel_time_s, after.average_travel_time_s, False),
        ("Total Waiting Time (s)", before.total_waiting_time_s, after.total_waiting_time_s, False),
        ("Average Queue Length (veh)", before.average_queue_length, after.average_queue_length, False),
        ("Vehicle Throughput (veh/h)", before.throughput_veh_per_hr, after.throughput_veh_per_hr, True),
        ("Bus Average Delay (s)", before.bus_average_delay_s, after.bus_average_delay_s, False),
        ("Passenger Total Delay (s)", before.passenger_total_delay_s, after.passenger_total_delay_s, False),
        ("CO2 Emissions (kg)", before.estimated_co2_kg, after.estimated_co2_kg, False),
        ("Fuel Consumption (L)", before.estimated_fuel_liters, after.estimated_fuel_liters, False),
    ]

    if before.emergency_response_time_s is not None and after.emergency_response_time_s is not None:
        comparison_items.append(
            ("Ambulance Response Time (s)", before.emergency_response_time_s, after.emergency_response_time_s, False)
        )

    table_rows = []
    structured_details = {}

    table_rows.append(f"{'Performance Metric':<32} | {label_before:<26} | {label_after:<26} | {'Delta (%)':<12} | {'Result':<12}")
    table_rows.append("-" * 115)

    for name, b_val, a_val, higher_better in comparison_items:
        pct, improved = calc_delta(b_val, a_val, higher_better)
        sign = "+" if pct > 0 else ""
        res_tag = "IMPROVED" if improved else ("NEUTRAL" if pct == 0 else "DEGRADED")
        table_rows.append(f"{name:<32} | {str(b_val):<26} | {str(a_val):<26} | {sign + str(pct) + '%':<12} | {res_tag:<12}")
        structured_details[name] = {
            "before": b_val,
            "after": a_val,
            "delta_pct": pct,
            "improved": improved
        }

    formatted_table = "\n".join(table_rows)

    return {
        "label_before": label_before,
        "label_after": label_after,
        "formatted_table": formatted_table,
        "metrics": structured_details
    }
