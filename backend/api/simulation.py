from fastapi import APIRouter, BackgroundTasks
import asyncio
from backend.orchestrator import orchestrator

router = APIRouter(prefix="/simulation", tags=["simulation"])

# Background task for auto simulation loop
_sim_task = None
_is_running = False

async def _simulation_loop():
    global _is_running
    while _is_running:
        orchestrator.cycle_step(force_optimize=False)
        await asyncio.sleep(1.0)

@router.get("/state")
def get_simulation_state():
    net_state = orchestrator.digital_twin.get_network_state()
    return {
        "simulation_step": orchestrator.digital_twin.simulation_step_count,
        "sim_time_sec": orchestrator.digital_twin.sim_time_sec,
        "is_running": _is_running,
        "network": net_state,
        "nodes": orchestrator.network.nodes_data,
        "vehicles": [v.dict() for v in orchestrator.digital_twin.vehicles.values()],
        "active_incidents": [inc.dict() for inc in orchestrator.digital_twin.active_incidents.values()],
        "signals": orchestrator.digital_twin.signal_phases,
        "charger_stations": orchestrator.ev_manager.stations
    }

@router.post("/start")
async def start_simulation():
    global _sim_task, _is_running
    if not _is_running:
        _is_running = True
        _sim_task = asyncio.create_task(_simulation_loop())
    return {"status": "started", "is_running": True}

@router.post("/stop")
async def stop_simulation():
    global _sim_task, _is_running
    _is_running = False
    if _sim_task:
        _sim_task.cancel()
        _sim_task = None
    return {"status": "stopped", "is_running": False}

@router.post("/step")
def step_simulation(force_optimize: bool = False):
    result = orchestrator.cycle_step(force_optimize=force_optimize)
    return result

@router.post("/reset")
def reset_simulation():
    global orchestrator
    orchestrator.digital_twin.active_incidents.clear()
    orchestrator.digital_twin._init_default_fleet()
    orchestrator.digital_twin.simulation_step_count = 0
    orchestrator.digital_twin.sim_time_sec = 0.0
    orchestrator.cycle_step(force_optimize=True)
    return {"status": "reset_completed"}
