from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.config import settings
from backend.api import (
    simulation_router,
    prediction_router,
    optimization_router,
    incidents_router,
    metrics_router,
)
from backend.orchestrator import orchestrator

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure baseline state is warm
    print(f"[{settings.PROJECT_NAME}] Starting up digital twin and AT-DQPSO optimizer...")
    yield
    # Shutdown
    print(f"[{settings.PROJECT_NAME}] Shutting down digital twin engine...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Predictive Multimodal Digital Twin for Adaptive Quantum-Inspired Urban Transportation Optimization",
    lifespan=lifespan
)

# Enable CORS for dashboard frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "sim_step": orchestrator.digital_twin.simulation_step_count,
        "active_vehicles": len(orchestrator.digital_twin.vehicles),
        "optimizer": "AT-DQPSO"
    }

# Include all modular routers
app.include_router(simulation_router)
app.include_router(prediction_router)
app.include_router(optimization_router)
app.include_router(incidents_router)
app.include_router(metrics_router)

# Also expose under /api prefix for versatility
app.include_router(simulation_router, prefix=settings.API_PREFIX)
app.include_router(prediction_router, prefix=settings.API_PREFIX)
app.include_router(optimization_router, prefix=settings.API_PREFIX)
app.include_router(incidents_router, prefix=settings.API_PREFIX)
app.include_router(metrics_router, prefix=settings.API_PREFIX)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
