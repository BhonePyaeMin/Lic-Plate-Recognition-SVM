import { useState } from 'react'
import { Columns3 } from 'lucide-react'
import Skeleton from '../../components/Skeleton.jsx'
import EngineBadge from '../../components/EngineBadge.jsx'

const SETS = [
  { key: 'edges', label: 'Compare edge detectors', ids: ['sobel', 'prewitt', 'canny'] },
  { key: 'segmentation', label: 'Compare segmentation', ids: ['otsu', 'region_growing', 'hsv_seg'] },
  { key: 'morphology', label: 'Compare morphology', ids: ['dilate', 'erode', 'open', 'close'] },
]

/** Runs several operations on the current crop and filter chain and shows them side by side. */
export default function ComparePanel({ prepare, disabled }) {
  const [state, setState] = useState({ phase: 'idle', title: '', tiles: [], error: null })
  const available = (ids) => ids.every((id) => prepare.byId[id])

  async function run(set) {
    setState({ phase: 'loading', title: set.label, tiles: set.ids.map((id) => ({ id })), error: null })
    try {
      const tiles = await Promise.all(
        set.ids.map(async (id) => {
          const params = Object.fromEntries(prepare.byId[id].params.map((p) => [p.key, p.default]))
          const res = await prepare.run([{ id, params }], { track: false })
          return { id, label: prepare.byId[id].label, image: res?.image, engine: res?.preprocessing.steps.at(-1)?.engine }
        }),
      )
      setState({ phase: 'ready', title: set.label, tiles, error: null })
    } catch (err) {
      setState({ phase: 'error', title: set.label, tiles: [], error: err.message })
    }
  }

  return (
    <section className="card compare" aria-labelledby="compare-title">
      <h3 id="compare-title" className="card__title">Compare methods</h3>
      <div className="compare__buttons">
        {SETS.map((s) => (
          <button key={s.key} type="button" className="btn btn--secondary btn--sm" disabled={disabled || !available(s.ids)} onClick={() => run(s)}>
            <Columns3 size={16} aria-hidden="true" /> {s.label}
          </button>
        ))}
      </div>
      {state.phase === 'error' && <p className="inline-note" role="alert">{state.error}</p>}
      {(state.phase === 'loading' || state.phase === 'ready') && (
        <div className="compare__grid" aria-busy={state.phase === 'loading'} data-testid="compare-grid">
          {state.tiles.map((t) => (
            <figure key={t.id} className="compare__tile">
              {t.image ? <img src={t.image} alt={`${t.label} result`} /> : <Skeleton width="100%" aspect="4 / 3" />}
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
