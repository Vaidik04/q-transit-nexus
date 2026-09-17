from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class VehicleType(str, Enum):
    PRIVATE = "private"
    DELIVERY = "delivery"
    BUS = "bus"
    EV = "ev"
    EMERGENCY = "emergency"

class VehicleState(BaseModel):
    vehicle_id: str
    type: VehicleType = VehicleType.DELIVERY
    current_edge: str
    origin: str
    destination: str
    capacity: int = 20
    priority: int = 1
    current_speed: float = 30.0
    route: List[str] = Field(default_factory=list)
    route_index: int = 0
    progress_on_edge: float = 0.0  # 0.0 to 1.0
    total_travel_time: float = 0.0
    delay: float = 0.0
    # Public transport fields
    passengers: int = 0
    line_id: Optional[str] = None
    target_headway: Optional[float] = None
    # EV fields
    soc: Optional[float] = None  # State of Charge 0.0 to 1.0
    battery_capacity_kwh: Optional[float] = None
    target_charger: Optional[str] = None
    # Emergency fields
    is_emergency_active: bool = False
