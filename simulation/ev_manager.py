"""
Module: simulation.ev_manager
Description:
    Manages Electric Vehicle (EV) state tracking, battery State of Charge (SoC %),
    dynamic power consumption modeling, and charging station replenishment.

What it reads:
    - TraCI vehicle telemetry (speed, acceleration, road position)
    - EV specifications from vehicle parameters

What it writes:
    - EVState records for canonical Digital Twin state engine
    - Battery degradation and charging session logs

Dependencies:
    - Depends on: traci_controller.py
    - Depended on by: state_manager.py, scenario_manager.py
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class EVProfile:
    vehicle_id: str
    battery_capacity_kwh: float
    battery_soc: float              # 0.0 to 100.0 %
    base_consumption_kwh_km: float   # kWh per km
    energy_consumed_kwh: float
    is_charging: bool
    current_charger_id: Optional[str]

class EVManager:
    """
    Models realistic EV battery drain, regenerative braking, and charging station interactions.
    """

    def __init__(self, controller):
        self.controller = controller
        self.ev_registry: Dict[str, EVProfile] = {}
        # Charging station lane positions
        self.charging_stations = {
            "cs_central_hub": {"lane": "E_11_21_0", "start": 180.0, "end": 210.0, "power_kw": 150.0},
            "cs_west_feeder": {"lane": "E_00_10_0", "start": 80.0, "end": 110.0, "power_kw": 50.0},
            "cs_north_depot": {"lane": "E_12_22_0", "start": 100.0, "end": 130.0, "power_kw": 100.0}
        }

    def register_ev(self, vehicle_id: str, capacity_kwh: float = 65.0, initial_soc: float = 80.0, base_cons: float = 0.18):
        self.ev_registry[vehicle_id] = EVProfile(
            vehicle_id=vehicle_id,
            battery_capacity_kwh=capacity_kwh,
            battery_soc=initial_soc,
            base_consumption_kwh_km=base_cons,
            energy_consumed_kwh=0.0,
            is_charging=False,
            current_charger_id=None
        )

    def step_ev_models(self, dt_seconds: float = 1.0):
        """Updates battery SoC and charging for all registered or discovered EVs."""
        traci = self.controller.traci
        if not self.controller.is_connected or not traci:
            return

        try:
            active_vehs = set(traci.vehicle.getIDList())

            # Auto-discover active EVs
            for vid in active_vehs:
                if vid not in self.ev_registry:
                    v_type = traci.vehicle.getTypeID(vid).lower()
                    if "ev" in v_type:
                        try:
                            init_soc = float(traci.vehicle.getParameter(vid, "battery_soc") or "80.0")
                            cap = float(traci.vehicle.getParameter(vid, "battery_capacity_kwh") or "65.0")
                            cons = float(traci.vehicle.getParameter(vid, "energy_consumption") or "0.18")
                        except Exception:
                            init_soc, cap, cons = 80.0, 65.0, 0.18
                        self.register_ev(vid, capacity_kwh=cap, initial_soc=init_soc, base_cons=cons)

            # Update battery dynamics for active EVs
            for vid, profile in list(self.ev_registry.items()):
                if vid not in active_vehs:
                    continue

                speed_ms = traci.vehicle.getSpeed(vid)
                accel = traci.vehicle.getAcceleration(vid)
                cur_lane = traci.vehicle.getLaneID(vid)
                pos = traci.vehicle.getLanePosition(vid)

                # Check if parked/charging at a designated charging station
                charging_now = False
                charger_found = None
                for cs_id, cs_data in self.charging_stations.items():
                    if cur_lane == cs_data["lane"] and (cs_data["start"] <= pos <= cs_data["end"]) and speed_ms < 0.2:
                        charging_now = True
                        charger_found = cs_id
                        # Charge replenishment: Energy (kWh) = Power (kW) * (dt / 3600)
                        added_kwh = cs_data["power_kw"] * (dt_seconds / 3600.0)
                        profile.battery_soc = min(100.0, profile.battery_soc + (added_kwh / profile.battery_capacity_kwh) * 100.0)
                        break

                profile.is_charging = charging_now
                profile.current_charger_id = charger_found

                if not charging_now:
                    # Dynamic energy consumption: distance (km) = (speed * dt) / 1000
                    distance_km = (speed_ms * dt_seconds) / 1000.0
                    # Acceleration penalty and regenerative braking credit
                    accel_factor = 1.0 + (accel * 0.15 if accel > 0 else accel * 0.08)
                    step_kwh = max(0.0001, distance_km * profile.base_consumption_kwh_km * max(0.2, accel_factor))

                    profile.energy_consumed_kwh += step_kwh
                    soc_drop = (step_kwh / profile.battery_capacity_kwh) * 100.0
                    profile.battery_soc = max(0.0, profile.battery_soc - soc_drop)

                # Sync back to TraCI parameter for GUI/teammate inspectability
                traci.vehicle.setParameter(vid, "battery_soc", f"{profile.battery_soc:.1f}")

        except Exception:
            pass

    def get_all_ev_states(self) -> List[Any]:
        """Returns EV state dataclasses compatible with state_manager."""
        from simulation.state_manager import EVState
        res = []
        for profile in self.ev_registry.values():
            res.append(EVState(
                vehicle_id=profile.vehicle_id,
                battery_soc=round(profile.battery_soc, 2),
                energy_consumed_kwh=round(profile.energy_consumed_kwh, 4),
                is_charging=profile.is_charging,
                charger_id=profile.current_charger_id
            ))
        return res
