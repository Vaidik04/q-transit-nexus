import os
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Q-Transit Nexus"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Simulation & Digital Twin Settings
    SIMULATION_STEP_SECONDS: float = 1.0
    OPTIMIZATION_INTERVAL_STEPS: int = 10
    DEFAULT_NETWORK_SCALE: str = "urban_corridor_grid"
    
    # Mode objective weight profiles
    # F = w1*T + w2*D + w3*C + w4*P + w5*E + w6*R + w7*V
    MODE_WEIGHTS: dict = {
        "balanced": {
            "w_time": 1.0,
            "w_dist": 0.4,
            "w_cong": 1.2,
            "w_pass": 1.5,
            "w_emis": 0.8,
            "w_reroute": 0.5,
            "w_viol": 50.0,
        },
        "emergency": {
            "w_time": 5.0,
            "w_dist": 0.1,
            "w_cong": 2.0,
            "w_pass": 0.2,
            "w_emis": 0.1,
            "w_reroute": 0.1,
            "w_viol": 100.0,
        },
        "green": {
            "w_time": 0.6,
            "w_dist": 0.8,
            "w_cong": 1.0,
            "w_pass": 0.5,
            "w_emis": 3.5,
            "w_reroute": 0.4,
            "w_viol": 50.0,
        },
        "transit": {
            "w_time": 0.8,
            "w_dist": 0.2,
            "w_cong": 1.0,
            "w_pass": 4.5,
            "w_emis": 0.6,
            "w_reroute": 0.8,
            "w_viol": 50.0,
        },
    }

    # AT-DQPSO hyper-parameters
    QPSO_ALPHA_START: float = 1.0
    QPSO_ALPHA_END: float = 0.5
    QPSO_DEFAULT_SWARM_SIZE: int = 30
    QPSO_DEFAULT_ITERATIONS: int = 40
    
    # Emission factor (approx kg CO2 per liter of fuel / per km)
    CO2_KG_PER_KM: float = 0.192
    IDLE_CO2_KG_PER_SEC: float = 0.0006

settings = Settings()
