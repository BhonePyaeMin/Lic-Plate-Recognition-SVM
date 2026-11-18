"""Domain errors mapped to safe, user-facing HTTP responses."""
from __future__ import annotations


class AppError(Exception):
    status_code = 500
    error_code = "INTERNAL_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class EmptyUploadError(AppError):
    status_code = 400
    error_code = "EMPTY_FILE"


class UnsupportedFileError(AppError):
    status_code = 415
    error_code = "UNSUPPORTED_FILE_TYPE"


class FileTooLargeError(AppError):
    status_code = 413
    error_code = "FILE_TOO_LARGE"


class InvalidImageError(AppError):
    status_code = 400
    error_code = "INVALID_IMAGE"


class InvalidParamsError(AppError):
    status_code = 400
    error_code = "INVALID_PARAMS"


class MatlabUnavailableError(AppError):
    status_code = 503
    error_code = "MATLAB_UNAVAILABLE"


class OcrUnavailableError(AppError):
    status_code = 503
    error_code = "OCR_UNAVAILABLE"


class SvmUnavailableError(AppError):
    status_code = 503
    error_code = "SVM_UNAVAILABLE"


class SvmOcrUnavailableError(AppError):
    status_code = 503
    error_code = "SVM_OCR_UNAVAILABLE"
