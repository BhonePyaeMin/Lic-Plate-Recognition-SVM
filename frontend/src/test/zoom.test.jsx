import { render, screen, waitFor, within, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../App.jsx'

const json = (body, status = 200) => Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) })
const NUM = (key, label, def, min, max, step = 1) => ({ key, label, kind: 'number', default: def, min, max, step })
const SELECT = (key, label, def, options) => ({ key, label, kind: 'select', default: def, options })
const CATALOG = {
  success: true,
  max_steps: 8,
  operations: [
    {
      id: 'scale', module: 3, group: 'Transformations', label: 'Scaling (zoom)', description: 'Enlarges or zooms in.', output: 'image', needs_reference: false, matlab: true,
      params: [
        NUM('factor', 'Factor', 1.5, 0.1, 4, 0.05),
        SELECT('mode', 'Mode', 'enlarge', ['enlarge', 'zoom-in']),
        NUM('center_x', 'Zoom-in centre x (%)', 50, 0, 100),
        NUM('center_y', 'Zoom-in centre y (%)', 50, 0, 100),
        SELECT('interpolation', 'Interpolation', 'bilinear', ['nearest', 'bilinear', 'bicubic']),
      ],
    },
  ],
}
const HEALTH = {
  status: 'ok', max_upload_mb: 10, allowed_extensions: ['jpg', 'png'],
  ocr: { available: true, enabled: true, state: 'ready', detail: 'ok' },
  matlab: { available: true, enabled: true, state: 'ready', detail: 'ok' },
}

class FakeImage {
  naturalWidth = 800
  naturalHeight = 600
  set src(_) { setTimeout(() => this.onload?.(), 0) }
}

function setup({ processedWidth = 800 } = {}) {
  vi.stubGlobal('Image', FakeImage)
  const fetchMock = vi.fn((url) => {
    const u = String(url)
    if (u.endsWith('/health')) return json(HEALTH)
    if (u.endsWith('/api/v1/filters')) return json(CATALOG)
    if (u.endsWith('/api/v1/filters/apply')) {
      return json({
        success: true, message: 'ok',
        result: {
          image: 'data:image/png;base64,AAAA', width: processedWidth, height: Math.round(processedWidth * 0.75), processing_time_ms: 5,
          preprocessing: { crop: null, steps: [{ id: 'scale', label: 'Scaling (zoom)', engine: 'matlab', fallback: false, duration_ms: 4, output: 'image' }] },
        },
      })
    }
    return json({ success: false, message: 'unexpected', error_code: 'X' })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}
const applyCalls = (f) => f.mock.calls.filter(([u]) => String(u).endsWith('/api/v1/filters/apply'))
const stepsOf = (call) => JSON.parse(call[1].body.get('steps'))

async function upload(user) {
  await user.upload(screen.getByLabelText('Choose a vehicle image'), new File([new Uint8Array(500)], 'car.jpg', { type: 'image/jpeg' }))
  await screen.findByRole('heading', { name: 'Filters' })
  await waitFor(() => expect(screen.getByLabelText('Choose a filter to add')).not.toBeDisabled())
}

describe('Compare interpolation', () => {
  it('runs the same zoom-in with nearest, bilinear and bicubic and shows three tiles', async () => {
    const fetchMock = setup()
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await user.click(screen.getByRole('button', { name: /Compare interpolation/ }))

    const grid = await screen.findByTestId('compare-grid')
    await waitFor(() => expect(within(grid).getAllByRole('img')).toHaveLength(3))
    const steps = applyCalls(fetchMock).map((c) => stepsOf(c).at(-1))
    expect(steps.map((s) => s.params.interpolation).sort()).toEqual(['bicubic', 'bilinear', 'nearest'])
    for (const s of steps) {
      expect(s.id).toBe('scale')
      expect(s.params).toMatchObject({ factor: 4, mode: 'zoom-in', center_x: 50, center_y: 50 })
    }
    expect(within(grid).getByAltText('nearest, zoom x4 result')).toBeInTheDocument()
    expect(within(grid).getByAltText('bicubic, zoom x4 result')).toBeInTheDocument()
  })

  it('uses the zoom factor and centre the user typed', async () => {
    const fetchMock = setup()
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    fireEvent.change(screen.getByLabelText('Zoom x'), { target: { value: '2.5' } })
    fireEvent.change(screen.getByLabelText('Centre x %'), { target: { value: '30' } })
    fireEvent.change(screen.getByLabelText('Centre y %'), { target: { value: '70' } })
    await user.click(screen.getByRole('button', { name: /Compare interpolation/ }))
    await screen.findByTestId('compare-grid')
    await waitFor(() => expect(applyCalls(fetchMock)).toHaveLength(3))
    for (const c of applyCalls(fetchMock)) {
      expect(stepsOf(c).at(-1).params).toMatchObject({ factor: 2.5, mode: 'zoom-in', center_x: 30, center_y: 70 })
    }
  })

  it('keeps out-of-range values inside the allowed limits', async () => {
    const fetchMock = setup()
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    fireEvent.change(screen.getByLabelText('Zoom x'), { target: { value: '9' } })
    fireEvent.change(screen.getByLabelText('Centre x %'), { target: { value: '250' } })
    await user.click(screen.getByRole('button', { name: /Compare interpolation/ }))
    await waitFor(() => expect(applyCalls(fetchMock)).toHaveLength(3))
    expect(stepsOf(applyCalls(fetchMock)[0]).at(-1).params).toMatchObject({ factor: 4, center_x: 100 })
  })
})

describe('Preview size', () => {
  async function addScale(user) {
    await user.selectOptions(screen.getByLabelText('Choose a filter to add'), 'scale')
    await user.click(screen.getByRole('button', { name: /Add/ }))
  }

  it('says how much bigger the processed image is and switches to actual size', async () => {
    setup({ processedWidth: 1640 }) // original is 800 px wide, so about 2.05x
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await addScale(user)
    expect(await screen.findByText(/Processed is 2\.05x the original size/)).toBeInTheDocument()
    expect(screen.getByText('800 x 600 px')).toBeInTheDocument()   // original size is shown too
    expect(screen.getByTestId('compare-pair')).not.toHaveClass('compare-pair--actual')

    await user.click(screen.getByRole('button', { name: 'Actual size (1:1)' }))
    expect(screen.getByTestId('compare-pair')).toHaveClass('compare-pair--actual')
    expect(screen.getByRole('button', { name: 'Actual size (1:1)' })).toHaveAttribute('aria-pressed', 'true')
    await user.click(screen.getByRole('button', { name: 'Fit to panel' }))
    expect(screen.getByTestId('compare-pair')).not.toHaveClass('compare-pair--actual')
  })

  it('does not mention a size difference when the image is unchanged in size', async () => {
    setup({ processedWidth: 800 })
    const user = userEvent.setup()
    render(<App />)
    await upload(user)
    await addScale(user)
    await screen.findByAltText('Processed image preview')
    expect(screen.queryByText(/Processed is .* the original size/)).not.toBeInTheDocument()
  })
})
