@echo off
rem Starts the whole app on this computer:
rem   1. the web page  (Docker container "plate-web", open http://localhost:8080)
rem   2. the backend   (port 8000; uses MATLAB automatically when it is installed)
rem Double-click this file. Keep this window open while you use the app; close it to stop the backend.
setlocal
cd /d "%~dp0"

echo [1/2] Starting the web page container...
docker start plate-web >nul 2>&1
if errorlevel 1 (
  echo       Could not start the container "plate-web". Open Docker Desktop, wait until it is running, then run this file again.
) else (
  echo       Web page ready: http://localhost:8080
)

echo [2/2] Starting the backend. The first start takes about 30 seconds because MATLAB starts too.
echo       When the page header shows "Service online, SVM ready" and "MATLAB: ready", it is working.
set CORS_ORIGINS=http://localhost:8080,http://127.0.0.1:8080
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
