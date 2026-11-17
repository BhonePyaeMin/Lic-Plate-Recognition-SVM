"""Response schemas. Every /api/v1 response uses one of these two envelopes."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class StepRecord(BaseModel):
    id: str
    label: str
    engine: Literal["opencv", "matlab"]
    fallback: bool = False  # MATLAB was preferred but failed, OpenCV ran instead
    duration_ms: float
    output: str


class PreprocessingInfo(BaseModel):
    crop: Optional[BoundingBox] = None
    steps: list[StepRecord] = []


class DetectionResult(BaseModel):
    # "recognized": text was read. "located_unreadable": a candidate region was
    # found but OCR returned nothing usable.
    detection_status: Literal["recognized", "located_unreadable"]
    plate_text: Optional[str] = None
    raw_ocr_text: Optional[str] = None
    confidence: Optional[float] = None  # 0..1, only when the OCR engine reports one
    bounding_box: BoundingBox
    detector: str  # identifier of the detector that produced the box
    detector_score: float  # heuristic ranking score, NOT a probability
    ocr_engine: Optional[str] = None
    image_width: int
    image_height: int
    processing_time_ms: float  # measured server side
    annotated_image: str  # data URL (JPEG)
    plate_crop: str  # data URL (PNG)
    matlab_used: bool = False
    preprocessing: Optional["PreprocessingInfo"] = None
    input_image: Optional[str] = None  # data URL of the image the detector actually saw (after crop and filters)


class DetectResponse(BaseModel):
    success: Literal[True] = True
    message: str
    result: DetectionResult


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    message: str
    error_code: str


class CapabilityStatus(BaseModel):
    available: bool
    enabled: bool = False
    state: str = ""
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    ocr: CapabilityStatus
    matlab: CapabilityStatus
    max_upload_mb: int
    allowed_extensions: list[str]


class ApplyResult(BaseModel):
    image: str  # data URL (JPEG, or PNG for binary masks)
    width: int
    height: int
    preprocessing: PreprocessingInfo
    processing_time_ms: float


class ApplyResponse(BaseModel):
    success: Literal[True] = True
    message: str
    result: ApplyResult


DetectionResult.model_rebuild()
