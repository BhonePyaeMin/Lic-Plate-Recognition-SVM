"""Bridge to MATLAB through the MATLAB Engine API for Python.

When MATLAB Engine is installed and ENABLE_MATLAB is true, the engine is
started in a background thread at server start. Operations from the shared
catalog are then executed by `matlab/api/run_operation_file.m`, using the same
operation ids and parameter keys as the OpenCV implementations. Images are
exchanged as PNG files in a private temporary directory that is deleted after
each call (the client filename is never used).

STATUS: exercised against MATLAB R2026a (all MATLAB-capable filters and one
detection ran through the Engine). The status reporting and the fallback to
OpenCV are also covered by tests. Other MATLAB releases were not tried.

To enable: install MATLAB (no toolbox needed), then
`pip install matlabengine` (version matching your MATLAB release) in the
backend virtual environment. Nothing else is required: the `matlab/` folder is
found automatically (override with MATLAB_PATH).
"""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.errors import MatlabUnavailableError
from app.schemas.detection import CapabilityStatus

_DEFAULT_MATLAB_DIR = Path(__file__).resolve().parents[3] / "matlab"


def _engine_package_present() -> bool:
    try:
        return importlib.util.find_spec("matlab.engine") is not None
    except (ImportError, ValueError):
        return False


class MatlabAdapter:
    def __init__(self, enabled: bool = False, matlab_path: Optional[str] = None, start_attempts: int = 3, retry_delay: float = 5.0):
        self.enabled = enabled
        self.start_attempts = max(1, start_attempts)  # MATLAB can fail to start once (memory, a slow previous session) and work a moment later
        self.retry_delay = retry_delay
        self.matlab_path = matlab_path or os.environ.get("MATLAB_PATH") or str(_DEFAULT_MATLAB_DIR)
        self._engine = None
        self._state = "disabled" if not enabled else ("starting" if _engine_package_present() else "no_package")
        self._error = ""
        self._lock = threading.Lock()  # one MATLAB call at a time

    # ------------------------------------------------------------ status
    @property
    def ready(self) -> bool:
        return self._state == "ready"

    def status(self) -> CapabilityStatus:
        detail = {
            "disabled": "MATLAB integration is disabled (ENABLE_MATLAB=false). OpenCV is used.",
            "no_package": "MATLAB Engine for Python is not installed, so OpenCV is used. Install MATLAB and `pip install matlabengine` to enable it.",
            "starting": "MATLAB is starting. OpenCV is used until it is ready.",
            "ready": "MATLAB is ready.",
            "failed": f"MATLAB could not be started ({self._error}). OpenCV is used.",
        }[self._state]
        return CapabilityStatus(available=self.ready, enabled=self.enabled, state=self._state, detail=detail)

    # ----------------------------------------------------------- startup
    def start_async(self) -> None:
        """Start the engine in the background (no-op unless enabled and installed)."""
        if self._state != "starting":
            return
        threading.Thread(target=self._start, name="matlab-start", daemon=True).start()

    def _start(self) -> None:
        # The state stays "starting" while retrying, so the UI never shows a failure that a
        # second attempt would have fixed.
        for attempt in range(1, self.start_attempts + 1):
            try:
                import matlab.engine  # type: ignore

                engine = matlab.engine.start_matlab()
                engine.addpath(engine.genpath(self.matlab_path), nargout=0)
                self._engine = engine
                self._state = "ready"
                return
            except Exception as exc:
                self._error = type(exc).__name__
                if attempt < self.start_attempts:
                    time.sleep(self.retry_delay)
        self._state = "failed"

    # --------------------------------------------------------------- run
    def run_operation(self, op_id: str, params: dict, image: np.ndarray, ref: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.ready:
            raise MatlabUnavailableError("MATLAB is not available on this server. Choose the OpenCV engine or install MATLAB Engine.")
        with self._lock, tempfile.TemporaryDirectory(prefix="lpr_matlab_") as tmp:
            src, dst, refp = (os.path.join(tmp, n) for n in ("in.png", "out.png", "ref.png"))
            cv2.imwrite(src, image)
            if ref is not None:
                cv2.imwrite(refp, ref)
            try:
                self._engine.run_operation_file(src, dst, op_id, json.dumps(params), refp if ref is not None else "", nargout=0)
            except Exception as exc:  # MatlabExecutionError and friends
                raise MatlabUnavailableError("MATLAB could not run this operation.") from exc
            out = cv2.imread(dst, cv2.IMREAD_UNCHANGED)
        if out is None:
            raise MatlabUnavailableError("MATLAB did not return an image.")
        if out.ndim == 3 and out.shape[2] == 4:
            out = out[:, :, :3]
        return out
