from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from contextlib import asynccontextmanager

from app.api.deps import get_matlab, get_settings
from app.api.routes import api_router, root_router
from app.core.errors import AppError

logger = logging.getLogger("lpr")


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"success": False, "message": message, "error_code": code})


def create_app() -> FastAPI:
    settings = get_settings()
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        get_matlab().start_async()  # no-op unless MATLAB Engine is installed and enabled
        yield

    app = FastAPI(title="License Plate Recognition API", version="1.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError):
        return _error(exc.status_code, exc.error_code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, __: RequestValidationError):
        return _error(400, "INVALID_REQUEST", "No image was received. Send the image as multipart form field 'file'.")

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        # Log type only: messages can include file content fragments. No traceback to the client.
        logger.error("Unhandled error: %s", type(exc).__name__, exc_info=True)
        return _error(500, "PROCESSING_FAILED", "Something went wrong while processing the image. Try again or use a different image.")

    app.include_router(root_router)
    app.include_router(api_router)
    return app


app = create_app()
