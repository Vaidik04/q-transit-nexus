from .simulation import router as simulation_router
from .prediction import router as prediction_router
from .optimization import router as optimization_router
from .incidents import router as incidents_router
from .metrics import router as metrics_router

__all__ = [
    "simulation_router",
    "prediction_router",
    "optimization_router",
    "incidents_router",
    "metrics_router",
]
