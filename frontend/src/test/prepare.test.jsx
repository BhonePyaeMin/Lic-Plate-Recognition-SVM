import { render, screen, waitFor, within, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../App.jsx'

const json = (body, status = 200) => Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) })
const op = (id, module, group, label, params = [], extra = {}) => ({
  id, module, group, label, description: `${label} description`, output: 'image', needs_reference: false, matlab: true, params, ...extra,
})
const CATALOG = {
  success: true,
  max_steps: 8,
  operations: [
    op('median', 1, 'Spatial filtering', 'Median smoothing', [{ key: 'kernel', label: 'Kernel size', kind: 'number', default: 3, min: 1, max: 31, step: 1 }]),
    op('gamma', 1, 'Histogram', 'Gamma correction', [{ key: 'gamma', label: 'Gamma', kind: 'number', default: 0.7, min: 0.1, max: 4, step: 0.05 }]),
    op('hist_equalization', 1, 'Histogram', 'Histogram equalization'),
    op('hist_matching', 1, 'Histogram', 'Histogram matching', [], { needs_reference: true }),
    op('sobel', 2, 'Edge detection', 'Sobel edges', [], { output: 'mask' }),
    op('prewitt', 2, 'Edge detection', 'Prewitt edges', [], { output: 'mask' }),
    op('canny', 2, 'Edge detection', 'Canny edges', [], { output: 'mask' }),
  ],
}
const health = (matlabState = 'no_package') => ({
  status: 'ok', max_upload_mb: 10, allowed_extensions: ['jpg', 'png'],
  ocr: { available: true, enabled: true, state: 'ready', detail: 'ok' },
  matlab: { available: matlabState === 'ready', enabled: true, state: matlabState, detail: matlabState },
})
const RESULT = {
  detection_status: 'recognized', plate_text: 'ABC1234', raw_ocr_text: 'ABC1234', confidence: 0.9,
  bounding_box: { x: 1, y: 2, width: 30, height: 10 }, detector: 'd', detector_score: 0.5, ocr_engine: 'o',
  image_width: 400, image_height: 300, processing_time_ms: 50, annotated_image: 'data:image/jpeg;base64,AA', plate_crop: 'data:image/png;base64,AA',
  matlab_used: true, input_image: 'data:image/jpeg;base64,CC',
  preprocessing: { crop: { x: 10, y: 20, width: 400, height: 300 }, steps: [{ id: 'median', label: 'Median smoothing', engine: 'matlab', fallback: false, duration_ms: 4, output: 'image' }] },
}
const applyResult = (engine = 'opencv', output = 'image') => ({
  success: true, message: 'ok',
  result: { image: 'data:image/png;base64,BBBB', width: 400, height: 300, processing_time_ms: 5,
    preprocessing: { crop: null, steps: [{ id: 'median', label: 'Median smoothing', engine, fallback: false, duration_ms: 3, output }] } },
})

class FakeImage {
  naturalWidth = 800
  naturalHeight = 600
  set src(_) { setTimeout(() => this.onload?.(), 0) }
}

function setup({ matlab = 'no_package', engineOnApply = 'opencv' } = {}) {
  vi.stubGlobal('Image', FakeImage)
  const fetchMock = vi.fn((url) => {
    const u = String(url)
    if (u.endsWith('/health')) return json(health(matlab))
    if (u.endsWith('/api/v1/filters')) return json(CATALOG)
    if (u.endsWith('/api/v1/filters/apply')) return json(applyResult(engineOnApply))
    return json({ success: true, message: 'License plate processed', result: RESULT })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}
const calls = (f, suffix) => f.mock.calls.filter(([u]) => String(u).endsWith(suffix))
const image = () => new File([new Uint8Array(500)], 'car.jpg', { type: 'image/jpeg' })

async function upload(user) {
  await user.upload(screen.getByLabelText('Choose a vehicle image'), image())
  await screen.findByRole('heading', { name: 'Filters' })
  await waitFor(() => expect(screen.getByLabelText('Choose a filter to add')).not.toBeDisabled())
}
async function addFilter(user, id) {
  await user.selectOptions(screen.getByLabelText('Choose a filter to add'), id)
  await user.click(screen.getByRole('button', { name: /Add/ }))
}

describe('white theme', () => {
  it('has a white page background and no dark-mode override', async () => {
    const { readFileSync } = await import('node:fs')
    const tokens = readFileSync('src/styles/tokens.css', 'utf8')
    expect(tokens).toMatch(/--bg:\s*#ffffff/i)
    expect(tokens).toMatch(/color-scheme:\s*light/)
    expect(tokens).not.toMatch(/prefers-color-scheme:\s*dark/)
  })
})

describe('prepare stage', () => {
  it('hides filters until an image is chosen, then lists catalog filters', async () => {
    setup()
    const user = userEvent.setup()
    render(<App />)
    expect(screen.queryByRole('heading', { name: 'Filters' })).not.toBeInTheDocument()
    await upload(user)
    const select = screen.getByLabelText('Choose a filter to add')
    expect(within(select).getByRole('option', { name: 'Median smoothing' })).toBeInTheDocument()
    expect(within(select).getByRole('option', { name: 'Sobel edges' })).toBeInTheDocument()
  })

  it('sends the filter chain, parameters and engine with detect', async () => {
    const fetchMock = setup()
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await addFilter(user, 'median')
    await addFilter(user, 'gamma')
    fireEvent.change(screen.getByLabelText('Kernel size', { selector: 'input[type=number]' }), { target: { value: '7' } })
    await user.selectOptions(screen.getByLabelText('Processing engine'), 'opencv')
    expect(screen.getByTestId('detect-summary')).toHaveTextContent('2 filters')

    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    await screen.findByTestId('result-view')
    const form = calls(fetchMock, '/api/v1/detect')[0][1].body
    expect(JSON.parse(form.get('steps'))).toEqual([
      { id: 'median', params: { kernel: 7 } },
      { id: 'gamma', params: { gamma: 0.7 } },
    ])
    expect(form.get('engine')).toBe('opencv')
    expect(form.get('crop')).toBeNull()
  })

  it('reorders and removes steps', async () => {
    const fetchMock = setup()
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await addFilter(user, 'median')
    await addFilter(user, 'gamma')
    await user.click(screen.getByRole('button', { name: 'Move Gamma correction up' }))
    await user.click(screen.getByRole('button', { name: 'Remove Median smoothing' }))
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    await screen.findByTestId('result-view')
    const steps = JSON.parse(calls(fetchMock, '/api/v1/detect')[0][1].body.get('steps'))
    expect(steps.map((s) => s.id)).toEqual(['gamma'])
  })

  it('sends a typed crop in original pixels', async () => {
    const fetchMock = setup()
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await waitFor(() => expect(screen.getByLabelText('Width')).not.toBeDisabled())
    fireEvent.change(screen.getByLabelText('X'), { target: { value: '100' } })
    fireEvent.change(screen.getByLabelText('Width'), { target: { value: '300' } })
    expect(await screen.findByTestId('crop-selection')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    await screen.findByTestId('result-view')
    expect(JSON.parse(calls(fetchMock, '/api/v1/detect')[0][1].body.get('crop'))).toEqual({ x: 100, y: 0, width: 300, height: 600 })
  })

  it('requires a reference image for filters that need one', async () => {
    setup()
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await addFilter(user, 'hist_matching')
    expect(screen.getByRole('button', { name: 'Detect plate' })).toBeDisabled()
    await user.upload(screen.getByLabelText(/Reference image/), new File([new Uint8Array(10)], 'ref.png', { type: 'image/png' }))
    expect(screen.getByRole('button', { name: 'Detect plate' })).not.toBeDisabled()
  })

  it('previews the processed image and shows which engine ran each step', async () => {
    const fetchMock = setup({ engineOnApply: 'matlab' })
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await addFilter(user, 'median')
    await user.click(screen.getByRole('button', { name: /Preview result/ }))
    expect(await screen.findByAltText('Processed image preview')).toBeInTheDocument()
    expect(calls(fetchMock, '/api/v1/filters/apply')).toHaveLength(1)
    expect(within(screen.getByLabelText('Engine used per step')).getByText('MATLAB')).toBeInTheDocument()
    expect(calls(fetchMock, '/api/v1/detect')).toHaveLength(0) // preview never runs detection
  })

  it('compares edge detectors with one request per method', async () => {
    const fetchMock = setup()
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await user.click(screen.getByRole('button', { name: 'Compare edge detectors' }))
    await waitFor(() => expect(screen.getAllByAltText(/ result$/)).toHaveLength(3))
    const sent = calls(fetchMock, '/api/v1/filters/apply').map(([, i]) => JSON.parse(i.body.get('steps'))[0].id)
    expect(sent.sort()).toEqual(['canny', 'prewitt', 'sobel'])
  })

  it('shows crop, filters and MATLAB use in the result', async () => {
    setup({ matlab: 'ready' })
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await user.click(screen.getByRole('button', { name: 'Detect plate' }))
    await screen.findByTestId('result-view')
    expect(screen.getByAltText('Image after crop and filters, as seen by the detector')).toBeInTheDocument()
    expect(screen.getByText('Used in 1 step(s)')).toBeInTheDocument()
    expect(screen.getByText('Crop applied')).toBeInTheDocument()
  })
})

describe('MATLAB status', () => {
  it.each([
    ['ready', 'MATLAB: ready'],
    ['no_package', 'MATLAB: not installed'],
    ['starting', 'MATLAB: starting'],
  ])('header shows %s state', async (state, text) => {
    setup({ matlab: state })
    render(<App />)
    expect(await screen.findByText(text)).toBeInTheDocument()
  })

  it('explains the fallback in the engine selector when MATLAB is missing', async () => {
    setup({ matlab: 'no_package' })
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    expect(screen.getByText(/MATLAB Engine is not installed on the server/)).toBeInTheDocument()
  })
})
