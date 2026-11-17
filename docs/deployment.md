# Deployment guide

The frontend and backend deploy separately. No hosting account was available while this was
written, so **nothing below has been deployed or verified on a host**. The local build and run
steps were executed; the container build was not.

## 1. Architecture of a deployment

```text
Browser --> static host (frontend/dist)        built with VITE_API_BASE_URL
        --> API host (backend container)       CORS_ORIGINS = the static host origin
```

The frontend is plain static files. The API is a Python process (about 1 GB RAM is a sensible
minimum for OpenCV plus the OCR runtime; measure on your host). Use HTTPS on both so the browser
allows the cross-origin call.

## 2. Backend

### Option 1: container (recommended for PaaS hosts)

```bash
cd backend
docker build -t plate-api .
docker run --rm -p 8000:8000 -e CORS_ORIGINS=https://YOUR-FRONTEND-ORIGIN plate-api
```

The Dockerfile has not been built yet; fix anything the first build reports. Most PaaS hosts
inject `PORT`, which the container honours.

### Option 2: virtual machine without Docker

```bash
cd backend
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
CORS_ORIGINS=https://YOUR-FRONTEND-ORIGIN uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Run it under a process manager (systemd, supervisor) behind a TLS-terminating reverse proxy.
Set the proxy's request body limit at or above `MAX_UPLOAD_MB` (nginx: `client_max_body_size`).

### Environment variables

See `.env.example`. Important in production:

* `CORS_ORIGINS`: only the exact frontend origin(s), no wildcard, no trailing slash.
* `MAX_UPLOAD_MB`: also reported to the UI through `/health`.
* `ENABLE_MATLAB`: harmless on hosts without MATLAB (it only looks for the Engine package).

The API stores nothing; no volume is required.

## 3. Frontend

```bash
cd frontend
npm ci
VITE_API_BASE_URL=https://YOUR-API-ORIGIN npm run build      # PowerShell: $env:VITE_API_BASE_URL="..."; npm run build
```

Upload the contents of `frontend/dist/` to any static host (Netlify, Vercel, Cloudflare Pages,
GitHub Pages, S3 plus CDN). Single page app, no client-side routes, so no rewrite rule is needed.
`VITE_API_BASE_URL` is baked in at build time; rebuild when the API URL changes.

## 4. Verification checklist

1. `curl https://YOUR-API-ORIGIN/health` returns `"status":"ok"` and `ocr.available: true`.
2. The frontend header shows "Service online, OCR ready".
3. Upload an image you may use; the result shows text, annotated image and crop.
4. Upload a `.txt` file and an oversized image; both show a clear error.
5. In browser dev tools, confirm the request to `/api/v1/detect` is not blocked by CORS.
6. Confirm no uploads are reachable by URL (the API has no static or upload routes).

## 5. MATLAB constraints

MATLAB cannot be assumed on a cloud host: it needs a licence and an installation, and the Engine
API for Python must match the MATLAB release. Recommended approach for the course:

* **Separate demonstration.** Run the MATLAB GUI locally and the web app from the deployed URLs.
* **Runtime integration.** Only on a machine with licensed MATLAB: run `pip install matlabengine`
  in the backend environment and restart. The backend starts MATLAB in the background and routes
  filters through `matlab/api/run_operation_file.m`. It was never run against real MATLAB, so test
  with engine "MATLAB only" before relying on it. On other hosts leave it as is: the header shows
  "MATLAB: not installed" and OpenCV runs everything.

## 6. Scaling notes

OCR and detection are CPU-bound and run in a worker thread; a single OCR engine instance is
guarded by a lock, so one process handles one recognition at a time. For concurrency run several
worker processes (`uvicorn --workers N`, each loads its own model) or several containers.
