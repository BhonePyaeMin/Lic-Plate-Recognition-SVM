import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../App.jsx'

const json = (body, status = 200) => Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) })
const CATALOG = {
  success: true,
  max_steps: 8,
  operations: [{ id: 'median', module: 1, group: 'Spatial filtering', label: 'Median smoothing', description: 'd', output: 'image', needs_reference: false, matlab: true, params: [] }],
}
const health = (svm, svmOcr) => ({
  status: 'ok', max_upload_mb: 10, allowed_extensions: ['jpg', 'png'],
  ocr: { available: true, enabled: true, state: 'ready', detail: 'ok' },
  matlab: { available: false, enabled: true, state: 'no_package', detail: 'none' },
  svm,
  svm_ocr: svmOcr,
})
const SVM_READY = { available: true, enabled: true, state: 'ready', detail: 'HOG + linear SVM, trained on synthetic scenes' }
const SVM_NONE = { available: false, enabled: true, state: 'no_model', detail: 'No trained model found.' }
const RESULT = (extra = {}) => ({
  detection_status: 'recognized', plate_text: 'ABC1234', raw_ocr_text: 'ABC1234', confidence: 0.9,
  bounding_box: { x: 1, y: 2, width: 30, height: 10 }, detector: 'opencv-contour-baseline+hog-svm', detector_score: 0.7, ocr_engine: 'o',
  image_width: 400, image_height: 300, processing_time_ms: 50, annotated_image: 'data:image/jpeg;base64,AA', plate_crop: 'data:image/png;base64,AA',
  matlab_used: false, svm_used: true, svm_score: 1.234, ...extra,
})

class FakeImage {
  naturalWidth = 800
  naturalHeight = 600
  set src(_) { setTimeout(() => this.onload?.(), 0) }
}

function setup(svm, result = RESULT(), svmOcr = undefined) {
  vi.stubGlobal('Image', FakeImage)
  const fetchMock = vi.fn((url) => {
    const u = String(url)
    if (u.endsWith('/health')) return json(health(svm, svmOcr))
    if (u.endsWith('/api/v1/filters')) return json(CATALOG)
    if (u.endsWith('/api/v1/filters/apply')) {
      return json({ success: true, message: 'ok', result: { image: 'data:image/png;base64,BB', width: 400, height: 300, processing_time_ms: 5, preprocessing: { crop: null, steps: [] } } })
    }
    return json({ success: true, message: 'License plate processed', result })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}
const calls = (f, suffix) => f.mock.calls.filter(([u]) => String(u).endsWith(suffix))

async function upload(user) {
  await user.upload(screen.getByLabelText('Choose a vehicle image'), new File([new Uint8Array(500)], 'car.jpg', { type: 'image/jpeg' }))
  await screen.findByRole('heading', { name: 'Filters' })
}

describe('SVM plate verification', () => {
  it('shows the SVM status in the header, and the new product name', async () => {
    setup(SVM_READY)
    render(<App />)
    expect(await screen.findByText('Service online, SVM ready')).toBeInTheDocument()
    expect(screen.getByText('License Plate Recognition Using SVM')).toBeInTheDocument()
    expect(screen.queryByText(/OCR ready/)).not.toBeInTheDocument()
  })

  it('reports an unavailable SVM and disables the SVM-only option', async () => {
    setup(SVM_NONE)
    const user = userEvent.setup()
    render(<App />)
    expect(await screen.findByText('Service online, SVM unavailable')).toBeInTheDocument()
    await upload(user)
    const select = screen.getByLabelText('Plate detection method')
    expect(within(select).getByRole('option', { name: /SVM verified/ })).toBeDisabled()
    expect(screen.getByText(/No SVM model is installed/)).toBeInTheDocument()
  })

  it('does not send detector_mode by default, and sends it when SVM verified is chosen', async () => {
    const fetchMock = setup(SVM_READY)
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await waitFor(() => expect(screen.getByLabelText('Plate detection method')).toBeEnabled())

    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    await screen.findByTestId('result-view')
    expect(calls(fetchMock, '/api/v1/detect')[0][1].body.get('detector_mode')).toBeNull()

    await user.selectOptions(screen.getByLabelText('Plate detection method'), 'svm')
    await user.click(screen.getByRole('button', { name: /Detect plate|Try again/ }))
    await waitFor(() => expect(calls(fetchMock, '/api/v1/detect')).toHaveLength(2))
    expect(calls(fetchMock, '/api/v1/detect')[1][1].body.get('detector_mode')).toBe('svm')
    // Choosing a detector must not trigger a filter preview request or carry the field there.
    for (const [, init] of calls(fetchMock, '/api/v1/filters/apply')) expect(init.body.get('detector_mode')).toBeNull()
  })

  it('shows the SVM score in the result, labelled as not a probability', async () => {
    setup(SVM_READY)
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    const view = await screen.findByTestId('result-view')
    expect(view).toHaveTextContent('SVM score')
    expect(view).toHaveTextContent('1.23 (plate side of the SVM boundary, not a probability)')
  })

  it('says the SVM was not applied when the result has no SVM score', async () => {
    setup(SVM_NONE, RESULT({ svm_used: false, svm_score: null, detector: 'opencv-contour-baseline' }))
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    const view = await screen.findByTestId('result-view')
    expect(view).toHaveTextContent(/SVM score\s*Not applied/)
  })
})

describe('live preview', () => {
  it('appears next to the controls after upload and shows the original until filters are added', async () => {
    setup(SVM_READY)
    const user = userEvent.setup()
    render(<App />)
    expect(screen.queryByRole('region', { name: 'Live preview' })).not.toBeInTheDocument()
    await upload(user)
    const live = await screen.findByRole('region', { name: 'Live preview' })
    expect(within(live).getByText('Original')).toBeInTheDocument()
    expect(within(live).getByAltText('Live preview of the current image')).toBeInTheDocument()
  })
})

describe('SVM character recognition', () => {
  const OCR_READY = { available: true, enabled: true, state: 'ready', detail: 'Character SVM (HOG)' }
  const OCR_NONE = { available: false, enabled: true, state: 'no_model', detail: 'No trained model found.' }
  const SVM_RESULT = RESULT({
    ocr_engine: 'svm-hog-chars', confidence: null,
    svm_chars: [{ char: 'A', margin: 2.34, x: 0, y: 0, width: 8, height: 20 }, { char: 'B', margin: 1.2, x: 10, y: 0, width: 8, height: 20 }],
  })

  it('disables the SVM engine when no character model is installed', async () => {
    setup(SVM_READY, RESULT(), OCR_NONE)
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    const select = screen.getByLabelText('Text recognition')
    expect(within(select).getByRole('option', { name: /SVM characters/ })).toBeDisabled()
    expect(screen.getByText(/No SVM character model is installed/)).toBeInTheDocument()
  })

  it('sends ocr_engine only when the SVM engine is chosen', async () => {
    const fetchMock = setup(SVM_READY, SVM_RESULT, OCR_READY)
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await waitFor(() => expect(screen.getByLabelText('Text recognition')).toBeEnabled())

    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    await screen.findByTestId('result-view')
    expect(calls(fetchMock, '/api/v1/detect')[0][1].body.get('ocr_engine')).toBeNull()

    await user.selectOptions(screen.getByLabelText('Text recognition'), 'svm')
    await user.click(screen.getByRole('button', { name: /Detect plate|Try again/ }))
    await waitFor(() => expect(calls(fetchMock, '/api/v1/detect')).toHaveLength(2))
    expect(calls(fetchMock, '/api/v1/detect')[1][1].body.get('ocr_engine')).toBe('svm')
    for (const [, init] of calls(fetchMock, '/api/v1/filters/apply')) expect(init.body.get('ocr_engine')).toBeNull()
  })

  it('lists each recognized character with its SVM margin', async () => {
    setup(SVM_READY, SVM_RESULT, OCR_READY)
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    const view = await screen.findByTestId('result-view')
    expect(view).toHaveTextContent('SVM characters')
    expect(view).toHaveTextContent('A (2.3)')
    expect(view).toHaveTextContent('B (1.2)')
    expect(view).toHaveTextContent('not probabilities')
    expect(view).not.toHaveTextContent(/OCR confidence/)
  })
})

describe('SVM recognition panel', () => {
  const OCR_READY = { available: true, enabled: true, state: 'ready', detail: 'Character SVM (HOG)' }
  const png = 'data:image/png;base64,AA'
  const cand = (index, status, text, extra = {}) => ({
    index, status, text, selected: false, source: 'contour', ocr_engine: 'svm-hog-chars', svm_score: 0.8,
    box: { x: 100 * index, y: 50, width: 120, height: 30 }, crop: png, segmented: null, glyphs: null, characters: null, ...extra,
  })
  const WITH_CANDIDATES = RESULT({
    ocr_engine: 'svm-hog-chars', confidence: null, image_width: 800, image_height: 600,
    candidates: [
      cand(1, 'readable', 'IF12YCC', { selected: true, segmented: png, glyphs: png, characters: 7 }),
      cand(2, 'small', null, { box: { x: 500, y: 300, width: 40, height: 12 } }),
    ],
  })

  async function detect(result) {
    setup(SVM_READY, result, OCR_READY)
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    await screen.findByTestId('result-view')
    return user
  }

  it('marks every candidate plate on the image and shows the chosen plate segmentation', async () => {
    await detect(WITH_CANDIDATES)
    const panel = screen.getByRole('region', { name: 'SVM recognition' })
    expect(within(panel).getByRole('button', { name: 'Plate 1: IF12YCC' })).toHaveAttribute('aria-pressed', 'true')
    expect(within(panel).getByRole('button', { name: 'Plate 2: not read' })).toHaveAttribute('aria-pressed', 'false')
    expect(within(panel).getByText(/segmented, 7 characters/)).toBeInTheDocument()
    expect(within(panel).getByAltText('Plate 1 with the characters found by the SVM boxed')).toBeInTheDocument()
    expect(within(panel).getByAltText('Plate 1 characters as the SVM classified them')).toBeInTheDocument()
  })

  it('lets the user inspect another plate by clicking it', async () => {
    const user = await detect(WITH_CANDIDATES)
    const panel = screen.getByRole('region', { name: 'SVM recognition' })
    await user.click(within(panel).getByRole('button', { name: 'Plate 2: not read' }))
    expect(within(panel).getByRole('button', { name: 'Plate 2: not read' })).toHaveAttribute('aria-pressed', 'true')
    expect(within(panel).getByRole('button', { name: 'Plate 1: IF12YCC' })).toHaveAttribute('aria-pressed', 'false')
    expect(within(panel).getByText('Small', { selector: '.chip' })).toBeInTheDocument()
    expect(within(panel).queryByAltText(/classified them/)).not.toBeInTheDocument()
  })

  it('says which engine read the plate and flags a fallback', async () => {
    await detect(WITH_CANDIDATES)
    expect(screen.getByTestId('result-view')).toHaveTextContent('Read by the SVM')
  })

  it('names the fallback engine when the SVM could not read the plate', async () => {
    await detect({ ...WITH_CANDIDATES, ocr_engine: 'rapidocr-onnxruntime', ocr_fallback: true })
    expect(screen.getByTestId('result-view')).toHaveTextContent('Read by RapidOCR (the SVM could not read this plate)')
  })

  it('hides the panel when the server returned no candidates', async () => {
    await detect(RESULT({ candidates: [] }))
    expect(screen.queryByRole('region', { name: 'SVM recognition' })).not.toBeInTheDocument()
  })
})

describe('cross-check chip', () => {
  const OCR_READY = { available: true, enabled: true, state: 'ready', detail: 'Character SVM (HOG)' }
  const base = RESULT({ ocr_engine: 'svm-hog-chars', confidence: null, plate_text: 'IF12YCC', candidates: [] })

  async function detect(result) {
    setup(SVM_READY, result, OCR_READY)
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    return screen.findByTestId('result-view')
  }

  it('says when RapidOCR agrees with the SVM', async () => {
    const view = await detect({ ...base, ocr_alternative: { engine: 'rapidocr-onnxruntime', text: 'IF12YCC', confidence: 0.98 } })
    expect(view).toHaveTextContent('RapidOCR agrees')
  })

  it('shows the other engine reading when it differs', async () => {
    const view = await detect({ ...base, ocr_alternative: { engine: 'rapidocr-onnxruntime', text: 'LTF12YCC', confidence: 0.9 } })
    expect(view).toHaveTextContent('RapidOCR read LTF12YCC (differs, check the plate)')
    expect(view).not.toHaveTextContent('RapidOCR agrees')
  })

  it('explains a hand-over caused by unreliable character splitting and shows the SVM reading', async () => {
    const view = await detect({
      ...base, ocr_engine: 'rapidocr-onnxruntime', ocr_fallback: true, ocr_fallback_reason: 'segmentation', plate_text: 'MH20EE0943',
      ocr_alternative: { engine: 'svm-hog-chars', text: 'MHNEEG94', confidence: null },
    })
    expect(view).toHaveTextContent("Read by RapidOCR (the SVM's character splitting looked unreliable)")
    expect(view).toHaveTextContent('The SVM read MHNEEG94 (differs, check the plate)')
  })

  it('shows nothing extra when there was no cross-check', async () => {
    const view = await detect(base)
    expect(view).not.toHaveTextContent(/RapidOCR (agrees|read)/)
  })
})
