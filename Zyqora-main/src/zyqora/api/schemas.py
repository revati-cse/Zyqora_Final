"""Pydantic request/response schemas for the Zyqora API."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ErrorInfo(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorBody(BaseModel):
    error: ErrorInfo


class Health(BaseModel):
    status: str
    version: str
    model_ready: bool
    model_code: Optional[str]
    profile: Optional[str]
    device_model: Optional[str]
    machine: str
    is_raspberry_pi: bool
    started_at: str
    offline_note: str = "All inference runs locally; no cloud services are used."


class EcoGuardOut(BaseModel):
    status: str
    threshold: float
    reasons: list[str]
    explanation: dict[str, str]


class SortOut(BaseModel):
    action: str
    actuate: bool
    message: str
    hardware: bool


class TimingOut(BaseModel):
    preprocess_ms: Optional[float]
    inference_ms: Optional[float]
    total_ms: Optional[float]


class ModelRef(BaseModel):
    profile: str
    description: str
    decoder: str
    pte: str
    sha256: Optional[str]
    runtime: str
    backend: str
    fallback_used: bool
    device: dict[str, Any]


class Classification(BaseModel):
    id: Optional[int]
    created_at: str
    filename: str
    source: str
    batch_id: Optional[str]
    predicted_category: Optional[str]
    confidence: Optional[float]
    class_scores: dict[str, float]
    score_note: str
    ecoguard: EcoGuardOut
    disposal_action: str
    guidance: dict[str, str]
    sort: SortOut
    timing: TimingOut
    model: ModelRef
    ground_truth: Optional[str]
    correct: Optional[bool]
    greenscore_event_id: Optional[int]
    image_url: Optional[str]
    capture_ms: Optional[float] = None


class BatchItemError(BaseModel):
    filename: str
    error: ErrorInfo


class BatchResponse(BaseModel):
    batch_id: str
    results: list[Classification]
    errors: list[BatchItemError]
    summary: dict[str, Any]


class HistoryPage(BaseModel):
    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


class CreateUser(BaseModel):
    user_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    display_name: Optional[str] = Field(default=None, max_length=64)


class VerifyRequest(BaseModel):
    event_id: int = Field(ge=1)
    observed_action: str = Field(min_length=2, max_length=32)
    correct_action: Optional[str] = Field(default=None, min_length=2, max_length=32)
    supervisor_id: Optional[str] = Field(default=None, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")


class VerifyResponse(BaseModel):
    event_id: int
    status: str
    observed_action: str
    correct_action: str
    awarded_points: int
    total_points: int


class VoicePlay(BaseModel):
    lang: str = Field(pattern=r"^(en|ta)$")
    key: str = Field(min_length=2, max_length=32)


class GreenScoreComponents(BaseModel):
    recycling_potential: float = Field(ge=0, le=100)
    environmental_impact: float = Field(ge=0, le=100)
    disposal_suitability: float = Field(ge=0, le=100)


class GreenScoreRequest(BaseModel):
    components: Optional[GreenScoreComponents] = None
    material: Optional[str] = Field(None, max_length=40)
    model_class: Optional[str] = Field(None, max_length=40)
    decision: Literal["ACCEPT", "REVIEW"] = "ACCEPT"
