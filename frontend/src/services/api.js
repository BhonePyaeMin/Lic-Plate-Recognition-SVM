const BASE = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '')

export const DEFAULT_LIMITS = { maxUploadMb: 10, allowedExtensions: ['jpg', 'jpeg', 'png', 'webp'] }

export class ApiError extends Error {
  constructor(code, message) {
    super(message)
    this.code = code
  }
}

/** Fetch backend capabilities. Returns null when the API cannot be reached. */
export async function getHealth(signal) {
  try {
    const res = await fetch(`${BASE}/health`, { signal })
    if (!res.ok) return null
    return await res.json()
  } catch {
    return null
  }
}

/** Fetch the filter catalog. Returns null when unavailable. */
export async function getFilters(signal) {
  try {
    const res = await fetch(`${BASE}/api/v1/filters`, { signal })
    if (!res.ok) return null
    return await res.json()
  } catch {
    return null
  }
}

/** Build the multipart body shared by preview and detect. */
export function buildForm(file, { crop, steps, engine, reference } = {}) {
  const body = new FormData()
  body.append('file', file)
  if (crop) body.append('crop', JSON.stringify(crop))
  if (steps?.length) body.append('steps', JSON.stringify(steps.map((s) => ({ id: s.id, params: s.params }))))
  body.append('engine', engine || 'auto')
  if (reference) body.append('reference', reference)
  return body
}

async function post(path, body, signal) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, { method: 'POST', body, signal })
  } catch (err) {
    if (err.name === 'AbortError') throw err
    throw new ApiError('API_UNAVAILABLE', 'The recognition service could not be reached. Check that the backend is running and try again.')
  }
  let json = null
  try {
    json = await res.json()
  } catch {
    /* non-JSON body handled below */
  }
  return { res, json }
}

function fail(json) {
  return new ApiError(json?.error_code ?? 'PROCESSING_FAILED', json?.message ?? 'The image could not be processed. Try again or use a different image.')
}

/**
 * Upload an image for detection.
 * Resolves to { found: true, message, result } or { found: false, message, code }.
 * Rejects with ApiError for every other failure.
 */
export async function detectPlate(file, options, signal) {
  const { res, json } = await post('/api/v1/detect', buildForm(file, options), signal)
  if (res.ok && json?.success) return { found: true, message: json.message, result: json.result }
  if (res.ok && json?.error_code === 'PLATE_NOT_FOUND') return { found: false, message: json.message, code: json.error_code }
  throw fail(json)
}

/** Crop and filter without detection. Resolves to the `result` object. */
export async function previewFilters(file, options, signal) {
  const { res, json } = await post('/api/v1/filters/apply', buildForm(file, options), signal)
  if (res.ok && json?.success) return json.result
  throw fail(json)
}
