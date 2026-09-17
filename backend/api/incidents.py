from fastapi import APIRouter, HTTPException
import time
from backend.orchestrator import orchestrator
from backend.schemas.incident import IncidentCreate, IncidentState, IncidentType

router = APIRouter(prefix="/incident", tags=["incidents"])

@router.post("")
def create_incident(inc_req: IncidentCreate = IncidentCreate(edge_id="E14")):
    """
    Simulates an incident (such as the default Road E14 accident)
    and executes the automated re-optimization cascade.
    """
    res = orchestrator.simulate_accident(edge_id=inc_req.edge_id, severity=inc_req.severity)
    return res

@router.get("/active")
def get_active_incidents():
    return list(orchestrator.digital_twin.active_incidents.values())

@router.post("/clear/{incident_id}")
def clear_incident(incident_id: str):
    if incident_id not in orchestrator.digital_twin.active_incidents:
        raise HTTPException(status_code=404, detail="Incident not found")
    orchestrator.digital_twin.clear_incident(incident_id)
    # Re-evaluate network
    orchestrator.cycle_step(force_optimize=True)
    return {"status": "cleared", "incident_id": incident_id}
