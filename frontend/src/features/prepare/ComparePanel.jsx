import { useState } from 'react'
import { Columns3, ZoomIn } from 'lucide-react'
import Skeleton from '../../components/Skeleton.jsx'
import EngineBadge from '../../components/EngineBadge.jsx'

const SETS = [
  { key: 'edges', label: 'Compare edge detectors', ids: ['sobel', 'prewitt', 'canny'] },
  { key: 'segmentation', label: 'Compare segmentation', ids: ['otsu', 'region_growing', 'hsv_seg'] },
  { key: 'morphology', label: 'Compare morphology', ids: ['dilate', 'erode', 'open', 'close'] },
]
const INTERPOLATIONS = ['nearest', 'bilinear', 'bicubic']

/**
 * Shows the zoomed picture at its real pixel size inside a fixed window, centred on the zoom point,
 * so differences between interpolation methods are visible instead of being shrunk away.
 */
function ZoomWindow({ src, alt, cx, cy }) {
  const [position, setPosition] = useState(`${cx}% ${cy}%`)
  function centre(e) {
    const img = e.currentTarget
    const box = img.parentElement
    // object-position p puts image point f*I at p*(C-I)+f*I; solve for the point to land mid-window.
    const axis = (f, image, window) => (image > window ? Math.min(100, Math.max(0, ((window / 2 - f * image) / (window - image)) * 100)) : 50)
    setPosition(`${axis(cx / 100, img.naturalWidth, box.clientWidth)}% ${axis(cy / 100, img.naturalHeight, box.clientHeight)}%`)
  }
  return (
    <div className="zoomwin">
      <img src={src} alt={alt} onLoad={centre} style={{ objectPosition: position }} />
    </div>
  )
}

function clamp(value, lo, hi) {
  const n = Number(value)
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : lo
}

/** Runs several operations on the current crop and filter chain and shows them side by side. */
export default function ComparePanel({ prepare, disabled }) {
  const [state, setState] = useState({ phase: 'idle', title: '', tiles: [], error: null, wide: false, centre: { x: 50, y: 50 } })
  const [zoom, setZoom] = useState({ factor: 4, x: 50, y: 50 })
  const available = (ids) => ids.every((id) => prepare.byId[id])

  /** jobs: [{ key, label, step: { id, params } }]. Each job runs the user's chain plus its own step. */
  async function runJobs(title, jobs, wide = false, centre = { x: 50, y: 50 }) {
    setState({ phase: 'loading', title, tiles: jobs.map((j) => ({ id: j.key })), error: null, wide, centre })
    try {
      const tiles = await Promise.all(
        jobs.map(async (j) => {
          const res = await prepare.run([j.step], { track: false })
          return { id: j.key, label: j.label, image: res?.image, engine: res?.preprocessing.steps.at(-1)?.engine }
        }),
      )
      setState({ phase: 'ready', title, tiles, error: null, wide, centre })
    } catch (err) {
      setState({ phase: 'error', title, tiles: [], error: err.message, wide, centre })
    }
  }

  function runSet(set) {
    const jobs = set.ids.map((id) => ({
      key: id,
      label: prepare.byId[id].label,
      step: { id, params: Object.fromEntries(prepare.byId[id].params.map((p) => [p.key, p.default])) },
    }))
    return runJobs(set.label, jobs)
  }

  /** Same zoom-in with each interpolation method: at high magnification the differences are visible. */
  function runInterpolation() {
    const f = clamp(zoom.factor, 1.5, 4)
    const jobs = INTERPOLATIONS.map((interpolation) => ({
      key: interpolation,
      label: `${interpolation}, zoom x${f}`,
      step: { id: 'scale', params: { factor: f, mode: 'zoom-in', center_x: clamp(zoom.x, 0, 100), center_y: clamp(zoom.y, 0, 100), interpolation } },
    }))
    return runJobs('Compare interpolation', jobs, true, { x: clamp(zoom.x, 0, 100), y: clamp(zoom.y, 0, 100) })
  }

  return (
    <section className="card compare" aria-labelledby="compare-title">
      <h3 id="compare-title" className="card__title">Compare methods</h3>
      <div className="compare__buttons">
        {SETS.map((s) => (
          <button key={s.key} type="button" className="btn btn--secondary btn--sm" disabled={disabled || !available(s.ids)} onClick={() => runSet(s)}>
            <Columns3 size={16} aria-hidden="true" /> {s.label}
          </button>
        ))}
        <button type="button" className="btn btn--secondary btn--sm" disabled={disabled || !prepare.byId.scale} onClick={runInterpolation}>
          <ZoomIn size={16} aria-hidden="true" /> Compare interpolation
        </button>
      </div>
      <div className="compare__zoomrow" role="group" aria-label="Interpolation comparison zoom">
        <span className="muted">Interpolation comparison magnifies a region (draw a crop around the plate first, or set the centre):</span>
        <label>Zoom x
          <input type="number" min="1.5" max="4" step="0.5" value={zoom.factor} onChange={(e) => setZoom({ ...zoom, factor: e.target.value })} />
        </label>
        <label>Centre x %
          <input type="number" min="0" max="100" step="5" value={zoom.x} onChange={(e) => setZoom({ ...zoom, x: e.target.value })} />
        </label>
        <label>Centre y %
          <input type="number" min="0" max="100" step="5" value={zoom.y} onChange={(e) => setZoom({ ...zoom, y: e.target.value })} />
        </label>
      </div>
      {state.phase === 'error' && <p className="inline-note" role="alert">{state.error}</p>}
      {(state.phase === 'loading' || state.phase === 'ready') && (
        <div className={`compare__grid${state.wide ? ' compare__grid--wide' : ''}`} aria-busy={state.phase === 'loading'} data-testid="compare-grid">
          {state.tiles.map((t) => (
            <figure key={t.id} className="compare__tile">
              {t.image
                ? (state.wide ? <ZoomWindow src={t.image} alt={`${t.label} result`} cx={state.centre.x} cy={state.centre.y} /> : <img src={t.image} alt={`${t.label} result`} />)
                : <Skeleton width="100%" aspect="4 / 3" />}
              <figcaption>
                <span>{t.label}</span>
                {t.engine && <EngineBadge engine={t.engine} />}
              </figcaption>
            </figure>
          ))}
        </div>
      )}
    </section>
  )
}
