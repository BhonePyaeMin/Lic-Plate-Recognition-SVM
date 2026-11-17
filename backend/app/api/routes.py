from __future__ import annotations

import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.api.deps import get_detector, get_matlab, get_recognizer, get_settings
from app.core.config import Settings
from app.core.errors import EmptyUploadError, FileTooLargeError
from app.schemas.detection import (
    ApplyResponse,
    ApplyResult,
    BoundingBox,
    CapabilityStatus,
    DetectResponse,
    ErrorResponse,
    HealthResponse,
    PreprocessingInfo,
    StepRecord,
)
from app.services.filters import CATALOG
from app.services.image_preprocessing import validate_and_decode
from app.services.matlab_adapter import MatlabAdapter
from app.services.pipeline import run_pipeline, to_data_url
from app.services.plate_detector import PlateDetector
from app.services.plate_recognizer import PlateRecognizer
from app.services.processing import ENGINES, MAX_STEPS, apply_crop, parse_crop, parse_engine, parse_steps, run_steps

root_router = APIRouter()
api_router = APIRouter(prefix="/api/v1")

_ERRORS = {code: {"model": ErrorResponse} for code in (400, 413, 415, 500, 503)}


@root_router.get("/health", response_model=HealthResponse)
def health(
    settings: Settings = Depends(get_settings),
    recognizer: PlateRecognizer = Depends(get_recognizer),
    matlab: MatlabAdapter = Depends(get_matlab),
) -> HealthResponse:
    ok, detail = recognizer.is_available()
    return HealthResponse(
        status="ok",
        ocr=CapabilityStatus(available=ok, enabled=True, state="ready" if ok else "failed", detail=detail),
        matlab=matlab.status(),
        max_upload_mb=settings.max_upload_mb,
        allowed_extensions=list(settings.allowed_extensions),
    )


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    """Read at most limit+1 bytes so an oversized upload is never fully buffered."""
    data = await file.read(limit + 1)
    if not data:
        raise EmptyUploadError("The uploaded file is empty. Choose an image and try again.")
    if len(data) > limit:
        raise FileTooLargeError("The image is larger than the upload limit. Resize or compress it and try again.")
    return data


def _prepare(
    image_bytes: bytes,
    filename: Optional[str],
    content_type: Optional[str],
    ref_bytes: Optional[bytes],
    ref_name: Optional[str],
    ref_type: Optional[str],
    crop_raw: Optional[str],
    steps_raw: Optional[str],
    engine_raw: Optional[str],
    settings: Settings,
    matlab: MatlabAdapter,
) -> tuple[np.ndarray, PreprocessingInfo]:
    """Validate, crop, then run the user's filter chain. CPU-bound: call from a worker thread."""
    engine = parse_engine(engine_raw)
    image = validate_and_decode(image_bytes, filename, content_type, settings)
    ref = validate_and_decode(ref_bytes, ref_name, ref_type, settings) if ref_bytes else None
    h, w = image.shape[:2]
    crop = parse_crop(crop_raw, w, h)
    steps = parse_steps(steps_raw)
    image = apply_crop(image, crop)
    image, records = run_steps(image, steps, engine, matlab, ref)
    info = PreprocessingInfo(
        crop=BoundingBox(x=crop[0], y=crop[1], width=crop[2], height=crop[3]) if crop else None,
        steps=[StepRecord(**r) for r in records],
    )
    return image, info


async def _inputs(file: UploadFile, reference: Optional[UploadFile], settings: Settings):
    data = await _read_limited(file, settings.max_upload_bytes)
    has_ref = reference is not None and bool(reference.filename)
    ref_data = await _read_limited(reference, settings.max_upload_bytes) if has_ref else None
    return data, ref_data, (reference.filename if has_ref else None), (reference.content_type if has_ref else None)


@api_router.get("/filters")
def filters():
    return {
        "success": True,
        "max_steps": MAX_STEPS,
        "engines": list(ENGINES),
        "operations": [op.public() for op in CATALOG.values()],
    }


@api_router.post("/filters/apply", response_model=ApplyResponse, responses=_ERRORS)
async def apply_filters(
    file: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
    crop: Optional[str] = Form(None),
    steps: Optional[str] = Form(None),
    engine: Optional[str] = Form("auto"),
    settings: Settings = Depends(get_settings),
    matlab: MatlabAdapter = Depends(get_matlab),
):
    """Preview: crop and filter an image without running detection."""
    started = time.perf_counter()
    data, ref_data, ref_name, ref_type = await _inputs(file, reference, settings)
    image, info = await run_in_threadpool(
        _prepare, data, file.filename, file.content_type, ref_data, ref_name, ref_type,
        crop, steps, engine, settings, matlab,
    )
    is_mask = bool(info.steps) and info.steps[-1].output == "mask"
    h, w = image.shape[:2]
    result = ApplyResult(
        image=to_data_url(image, ".png" if is_mask else ".jpg"),
        width=w, height=h, preprocessing=info,
        processing_time_ms=round((time.perf_counter() - started) * 1000.0, 1),
    )
    return ApplyResponse(message="Preview ready", result=result)


@api_router.post("/detect", response_model=DetectResponse, responses=_ERRORS)
async def detect(
    file: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
    crop: Optional[str] = Form(None),
    steps: Optional[str] = Form(None),
    engine: Optional[str] = Form("auto"),
    settings: Settings = Depends(get_settings),
    detector: PlateDetector = Depends(get_detector),
    recognizer: PlateRecognizer = Depends(get_recognizer),
    matlab: MatlabAdapter = Depends(get_matlab),
):
    data, ref_data, ref_name, ref_type = await _inputs(file, reference, settings)
    # Decoding, filtering and OCR are CPU-bound; keep them off the event loop.
    image, info = await run_in_threadpool(
        _prepare, data, file.filename, file.content_type, ref_data, ref_name, ref_type,
        crop, steps, engine, settings, matlab,
    )
    result = await run_in_threadpool(run_pipeline, image, detector, recognizer, settings.ocr_candidates, info)
    if result is None:
        return JSONResponse(
            status_code=200,
            content=ErrorResponse(
                message="No license plate detected. Try a clearer, closer photo with the plate facing the camera, or adjust the crop and filters.",
                error_code="PLATE_NOT_FOUND",
            ).model_dump(),
        )
    if result.detection_status == "recognized":
        message = "License plate processed"
    else:
        message = "A possible plate region was found, but its text could not be read."
    return DetectResponse(message=message, result=result)
