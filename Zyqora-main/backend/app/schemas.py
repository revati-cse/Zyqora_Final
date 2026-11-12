from enum import Enum
from pydantic import BaseModel
from typing import Optional

class ModeEnum(str, Enum):
    LIVE = "LIVE"
    REPLAY = "REPLAY"
    DEMO = "DEMO"

class ItemResult(BaseModel):
    filename: str
    mode: ModeEnum
    model_class: Optional[str] = None
    confidence: float = 0.0
    category: Optional[str] = None
    trust: str = "REVIEW"
    action: str = "CHECK"
    retried: bool = False
    latency_ms: float = 0.0
    note: Optional[str] = None

class Telemetry(BaseModel):
    mode: ModeEnum
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    temperature_c: Optional[float] = None
    device: str
