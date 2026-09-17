from typing import Optional
from pydantic import BaseModel, Field

class SimulationMetrics(BaseModel):
    traffic: str = Field("35%", description="Traffic density percentage or summary")
    congestion: float = Field(0.35, description="Average network congestion ratio between 0.0 and 1.0")
    vehicle_count: int = Field(25, description="Total active vehicles in digital twin")
    average_speed: float = Field(38.5, description="Network average speed in km/h")
    bus_delay: float = Field(1.2, description="Average bus delay in minutes")
    passenger_delay: float = Field(18.4, description="Cumulative passenger delay in person-minutes")
    co2: float = Field(14.8, description="Estimated total CO2 emissions in kg or savings percentage")
    co2_savings_percent: Optional[float] = Field(12.4, description="CO2 emissions reduction vs uncoordinated baseline")
    active_incidents: int = Field(0, description="Number of currently active roadway incidents")
    timestamp: Optional[float] = Field(None, description="Simulation timestamp in seconds")
