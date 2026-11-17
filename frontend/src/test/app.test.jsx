import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../App.jsx'
import { validateFile } from '../utils/validateFile'

const HEALTH = {
  status: 'ok',
  max_upload_mb: 10,
  allowed_extensions: ['jpg', 'jpeg', 'png', 'webp'],
  ocr: { available: true, enabled: true, detail: 'ok' },
  matlab: { available: false, enabled: false, detail: 'disabled' },
}

const RESULT = {
  detection_status: 'recognized',
  plate_text: 'XYZ9876',
  raw_ocr_text: 'XYZ 9876',
  confidence: 0.93,
  bounding_box: { x: 10, y: 20, width: 200, height: 50 },
  detector: 'test-detector',
  detector_score: 0.71,
  ocr_engine: 'test-ocr',
  image_width: 800,
  image_height: 600,
  processing_time_ms: 123.4,
  annotated_image: 'data:image/jpeg;base64,AAAA',
  plate_crop: 'data:image/png;base64,AAAA',
  matlab_used: false,
}

const json = (body, status = 200) => Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) })

const op = (id, module, group, label, params = [], extra = {}) => ({ id, module, group, label, description: label + ' description', output: 'image', needs_reference: false, matlab: true, params, ...extra })
const CATALOG = {
  success: true,
  max_steps: 8,
  engines: ['auto', 'opencv', 'matlab'],
  operations: [
    op('median', 1, 'Spatial filtering', 'Median smoothing', [{ key: 'kernel', label: 'Kernel size', kind: 'number', default: 3, min: 1, max: 31, step: 1 }]),
    op('hist_matching', 1, 'Histogram', 'Histogram matching', [], { needs_reference: true }),
    op('sobel', 2, 'Edge detection', 'Sobel edges', [], { output: 'mask' }),
    op('prewitt', 2, 'Edge detection', 'Prewitt edges', [], { output: 'mask' }),
    op('canny', 2, 'Edge detection', 'Canny edges', [], { output: 'mask' }),
    op('rotate', 3, 'Transformations', 'Rotation', [{ key: 'interpolation', label: 'Interpolation', kind: 'select', default: 'bilinear', options: ['nearest', 'bilinear', 'bicubic'] }]),
  ],
}
const APPLY = (engine = 'opencv') => ({
  success: true,
  message: 'Preview ready',
  result: {
    image: 'data:image/png;base64,BBBB', width: 400, height: 300, processing_time_ms: 5,
    preprocessing: { crop: null, steps: [{ id: 'median', label: 'Median smoothing', engine, fallback: false, duration_ms: 3, output: 'image' }] },
  },
})

/** detectImpl returns the promise for POST /api/v1/detect. */
function mockApi(detectImpl, { health = HEALTH } = {}) {
  const fetchMock = vi.fn((url, init) => {
    const u = String(url)
    if (u.endsWith('/health')) return json(health)
    if (u.endsWith('/api/v1/filters')) return json(CATALOG)
    if (u.endsWith('/api/v1/filters/apply')) return json(APPLY())
    return detectImpl(url, init)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

const detectCalls = (f) => f.mock.calls.filter(([u]) => String(u).includes('/api/v1/detect'))
const image = (name = 'car.jpg', type = 'image/jpeg', size = 1000) =>
  new File([new Uint8Array(size)], name, { type })

async function chooseFile(user, file) {
  await user.upload(screen.getByLabelText('Choose a vehicle image'), file)
}

describe('upload validation', () => {
  it('rejects unsupported file types without calling the API', async () => {
    const fetchMock = mockApi(() => json({}))
    const user = userEvent.setup({ applyAccept: false })
    render(<App />)
    await chooseFile(user, image('notes.txt', 'text/plain'))
    expect(await screen.findByRole('alert')).toHaveTextContent(/not supported/i)
    expect(screen.getByRole('heading', { name: 'Unsupported file type' })).toBeInTheDocument()
    expect(detectCalls(fetchMock)).toHaveLength(0)
  })

  it('rejects oversized files using the backend limit', async () => {
    mockApi(() => json({}))
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getByText(/Up to 10 MB/)).toBeInTheDocument())
    await chooseFile(user, image('big.jpg', 'image/jpeg', 11 * 1024 * 1024))
    expect(await screen.findByRole('alert')).toHaveTextContent(/larger than 10 MB/)
  })

  it('unit: validateFile accepts a valid file', () => {
    const limits = { maxUploadMb: 10, allowedExtensions: ['jpg', 'png'] }
    expect(validateFile(image('a.PNG', 'image/png'), limits)).toBeNull()
    expect(validateFile(image('a.gif', 'image/gif'), limits).code).toBe('UNSUPPORTED_FILE_TYPE')
    expect(validateFile(image('a.jpg', 'image/jpeg', 0), limits).code).toBe('EMPTY_FILE')
  })
})

describe('detection flow', () => {
  it('does not process until the user asks, then shows skeleton then result', async () => {
    let resolve
    const fetchMock = mockApi(() => new Promise((r) => { resolve = r }))
    const user = userEvent.setup()
    render(<App />)
    await chooseFile(user, image())

    expect(screen.getByText('Ready to analyze')).toBeInTheDocument()
    expect(detectCalls(fetchMock)).toHaveLength(0)

    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    expect(screen.getByTestId('result-skeleton')).toBeInTheDocument()
    expect(screen.getAllByText('Analyzing image').length).toBeGreaterThan(0)
    expect(screen.getByRole('region', { name: 'Results' })).toHaveAttribute('aria-busy', 'true')
    expect(detectCalls(fetchMock)).toHaveLength(1)

    resolve({ ok: true, status: 200, json: () => Promise.resolve({ success: true, message: 'License plate processed', result: RESULT }) })
    expect(await screen.findByTestId('result-view')).toBeInTheDocument()
    expect(screen.queryByTestId('result-skeleton')).not.toBeInTheDocument()
    expect(screen.getByText('XYZ9876')).toBeInTheDocument()
    expect(screen.getByText('OCR confidence 93%')).toBeInTheDocument()
    expect(screen.getByText('123 ms')).toBeInTheDocument()
    expect(screen.getByText('Not used')).toBeInTheDocument() // MATLAB
    expect(screen.getByAltText('Image with the detected plate outlined')).toBeInTheDocument()
  })

  it('hides confidence when the backend returns none', async () => {
    mockApi(() => json({ success: true, message: 'ok', result: { ...RESULT, confidence: null } }))
    const user = userEvent.setup()
    render(<App />)
    await chooseFile(user, image())
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    await screen.findByTestId('result-view')
    expect(screen.queryByText(/OCR confidence/)).not.toBeInTheDocument()
  })

  it('copies plate text', async () => {
    mockApi(() => json({ success: true, message: 'ok', result: RESULT }))
    const user = userEvent.setup()
    render(<App />)
    await chooseFile(user, image())
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    await screen.findByTestId('result-view')
    await user.click(screen.getByRole('button', { name: 'Copy plate text' }))
    expect(await screen.findByText('Copied')).toBeInTheDocument()
    expect(await navigator.clipboard.readText()).toBe('XYZ9876')
  })

  it('shows the no-plate state without any fake text', async () => {
    mockApi(() => json({ success: false, message: 'No license plate detected.', error_code: 'PLATE_NOT_FOUND' }))
    const user = userEvent.setup()
    render(<App />)
    await chooseFile(user, image())
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    expect(await screen.findByRole('heading', { name: 'No license plate detected' })).toBeInTheDocument()
    expect(screen.queryByTestId('result-view')).not.toBeInTheDocument()
  })

  it('shows an unreadable-plate result without plate text', async () => {
    mockApi(() => json({ success: true, message: 'x', result: { ...RESULT, detection_status: 'located_unreadable', plate_text: null, confidence: null } }))
    const user = userEvent.setup()
    render(<App />)
    await chooseFile(user, image())
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    expect(await screen.findByText(/text could not be read/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Copy plate text' })).not.toBeInTheDocument()
  })

  it('shows a clear error when the API is unreachable and allows retry', async () => {
    let fail = true
    mockApi(() => (fail ? Promise.reject(new TypeError('Failed to fetch')) : json({ success: true, message: 'ok', result: RESULT })))
    const user = userEvent.setup()
    render(<App />)
    await chooseFile(user, image())
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    expect(await screen.findByRole('heading', { name: 'Service unavailable' })).toBeInTheDocument()
    fail = false
    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByTestId('result-view')).toBeInTheDocument()
  })

  it('surfaces server error messages', async () => {
    mockApi(() => json({ success: false, message: 'The image data is corrupted and could not be decoded. Try another file.', error_code: 'INVALID_IMAGE' }, 400))
    const user = userEvent.setup()
    render(<App />)
    await chooseFile(user, image())
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/corrupted/)
  })
})

describe('layout', () => {
  it('uses responsive breakpoints and reduced-motion handling in the stylesheet', async () => {
    const { readFileSync } = await import('node:fs')
    const css = readFileSync('src/styles/components.css', 'utf8')
    const base = readFileSync('src/styles/base.css', 'utf8')
    expect(css).toMatch(/@media \(max-width: 960px\)/)
    expect(css).toMatch(/@media \(max-width: 640px\)/)
    expect(css + base).toMatch(/prefers-reduced-motion: reduce/)
  })
})
