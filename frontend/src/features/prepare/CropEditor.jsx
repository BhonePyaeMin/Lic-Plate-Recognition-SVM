import { useRef, useState } from 'react'
import { Crop, Maximize } from 'lucide-react'

const MIN_W = 24
const MIN_H = 12

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi)

/**
 * Draw a crop rectangle on the image with the pointer, or type exact pixel
 * values. Coordinates are always in original-image pixels.
 */
export default function CropEditor({ src, dimensions, crop, onChange }) {
  const boxRef = useRef(null)
  const drag = useRef(null)
  const [draft, setDraft] = useState(null) // fractional rect while dragging
  const nw = dimensions?.width
  const nh = dimensions?.height

  const point = (e) => {
    const r = boxRef.current.getBoundingClientRect()
    return { x: clamp((e.clientX - r.left) / r.width, 0, 1), y: clamp((e.clientY - r.top) / r.height, 0, 1) }
  }
  const toRect = (a, b) => ({ x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) })

  function down(e) {
    if (!nw) return
    e.currentTarget.setPointerCapture(e.pointerId)
    drag.current = point(e)
    setDraft({ ...drag.current, w: 0, h: 0 })
  }
  function move(e) {
    if (drag.current) setDraft(toRect(drag.current, point(e)))
  }
  function up(e) {
    if (!drag.current) return
    const r = toRect(drag.current, point(e))
    drag.current = null
    setDraft(null)
    const next = { x: Math.round(r.x * nw), y: Math.round(r.y * nh), width: Math.round(r.w * nw), height: Math.round(r.h * nh) }
    if (next.width >= MIN_W && next.height >= MIN_H) onChange(next)
  }

  function setField(key, raw) {
    if (!nw) return
    const base = crop ?? { x: 0, y: 0, width: nw, height: nh }
    const v = Math.round(Number(raw))
    if (!Number.isFinite(v)) return
    const next = { ...base, [key]: v }
    next.x = clamp(next.x, 0, nw - 1)
    next.y = clamp(next.y, 0, nh - 1)
    next.width = clamp(next.width, 1, nw - next.x)
    next.height = clamp(next.height, 1, nh - next.y)
    onChange(next)
  }

  const shown = draft ?? (crop && nw ? { x: crop.x / nw, y: crop.y / nh, w: crop.width / nw, h: crop.height / nh } : null)
  const invalid = crop && (crop.width < MIN_W || crop.height < MIN_H)

  return (
    <div className="crop">
      <div
        className="crop__stage"
        ref={boxRef}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerCancel={() => { drag.current = null; setDraft(null) }}
        style={nw ? { aspectRatio: `${nw} / ${nh}` } : undefined}
      >
        <img src={src} alt="Vehicle image. Drag on it to select the crop area." draggable={false} />
        {shown && (
          <div
            className="crop__sel"
            data-testid="crop-selection"
            style={{ left: `${shown.x * 100}%`, top: `${shown.y * 100}%`, width: `${shown.w * 100}%`, height: `${shown.h * 100}%` }}
          />
        )}
      </div>

      <p className="muted crop__hint">
        <Crop size={14} aria-hidden="true" /> Drag on the image to select the area to analyze, or type exact values.
      </p>
      <fieldset className="crop__fields" disabled={!nw}>
        <legend className="sr-only">Crop area in pixels</legend>
        {[['x', 'X'], ['y', 'Y'], ['width', 'Width'], ['height', 'Height']].map(([k, label]) => (
          <label key={k} className="field">
            <span>{label}</span>
            <input
              type="number"
              min="0"
              inputMode="numeric"
              value={crop ? crop[k] : k === 'width' ? nw ?? '' : k === 'height' ? nh ?? '' : 0}
              onChange={(e) => setField(k, e.target.value)}
            />
          </label>
        ))}
      </fieldset>
      {invalid && <p className="inline-note" role="alert">Crop must be at least {MIN_W} x {MIN_H} px.</p>}
      <div className="crop__actions">
        <button type="button" className="btn btn--ghost btn--sm" onClick={() => onChange(null)} disabled={!crop}>
          <Maximize size={16} aria-hidden="true" /> Use full image
        </button>
        {crop && nw && <span className="muted">{crop.width} x {crop.height} px selected</span>}
      </div>
    </div>
  )
}
