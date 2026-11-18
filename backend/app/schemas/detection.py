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


class SvmChar(BaseModel):
    char: str
    margin: float  # SVM score of the chosen class: distance-like, NOT a probability
    x: int
    y: int
    width: int
    height: int


class CandidateInfo(BaseModel):
    """One candidate plate region that was read, for the SVM view (best one is `selected`)."""
    index: int  # 1-based, in ranking order
    box: BoundingBox  # coordinates of the analysed image, like DetectionResult.bounding_box
    source: str
    status: Literal["readable", "small", "unread"]  # readable: text found; small: tiny region; unread: no usable text
    text: Optional[str] = None
    selected: bool = False
    ocr_engine: Optional[str] = None
    svm_score: Optional[float] = None  # plate-vs-not SVM margin, None when the SVM verifier did not run
    characters: Optional[int] = None  # glyphs the character SVM segmented
    crop: str  # data URL (PNG), the deskewed plate crop
    segmented: Optional[str] = None  # data URL (PNG): plate with the SVM's character boxes
    glyphs: Optional[str] = None  # data URL (PNG): normalised glyphs with predicted characters


class OcrAlternative(BaseModel):
    """The other engine's reading of the same plate, shown so a disagreement is visible."""
    engine: Optional[str] = None
    text: Optional[str] = None
    confidence: Optional[float] = None


class DetectionResult(BaseModel):
    # "recognized": text was read. "located_unreadable": a candidate region was
    # found but OCR returned nothing usable.
    detection_status: Literal["recognized", "located_unreadable"]
    plate_text: Optional[str] = None
    raw_ocr_text: Optional[str] = None
    confidence: Optional[float] = None  # 0..1, only when the OCR engine reports one
    bounding_box: BoundingBox
    detector: str  # identifier of the detector that produced the box
    detector_score: float  # ranking score (blend when the SVM ran), NOT a probability
    svm_used: bool = False  # the HOG + SVM verifier scored this region
    svm_score: Optional[float] = None  # SVM margin: distance-like, positive = plate side. NOT a probability
    ocr_engine: Optional[str] = None
    ocr_fallback: bool = False  # the SVM could not read the plate, so RapidOCR did
    ocr_fallback_reason: Optional[str] = None  # 'unreadable' (SVM found no text) or 'segmentation' (SVM splitting unreliable)
    ocr_alternative: Optional[OcrAlternative] = None  # cross-check by the second engine (Automatic mode)
    candidates: list[CandidateInfo] = []
    svm_chars: Optional[list[SvmChar]] = None  # per-character SVM output, only when ocr_engine is the SVM
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
    svm: Optional[CapabilityStatus] = None
    svm_ocr: Optional[CapabilityStatus] = None
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
