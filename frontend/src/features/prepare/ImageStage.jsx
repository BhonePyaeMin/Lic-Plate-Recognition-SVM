import { useId, useState } from 'react'
import { Eye } from 'lucide-react'
import CropEditor from './CropEditor.jsx'
import Skeleton from '../../components/Skeleton.jsx'
import EngineBadge from '../../components/EngineBadge.jsx'

/** Tabs: select the crop on the original, or preview the processed result next to the original. */
export default function ImageStage({ detection, prepare }) {
  const { previewUrl, dimensions, file } = detection
  const { preview } = prepare
  const [tab, setTab] = useState('crop')
  const base = useId()
  const hasChanges = Boolean(prepare.crop) || prepare.steps.length > 0

  async function runPreview() {
    setTab('preview')
    try { await prepare.run() } catch { /* shown via preview.error */ }
  }

  const aspect = dimensions ? `${dimensions.width} / ${dimensions.height}` : '4 / 3'

  return (
    <section className="card stage" aria-labelledby={`${base}-title`}>
      <div className="stage__head">
        <h2 id={`${base}-title`} className="card__title">Prepare image</h2>
        <div role="tablist" aria-label="Prepare views" className="tabs">
          {[['crop', 'Crop'], ['preview', 'Preview']].map(([id, label]) => (
            <button key={id} role="tab" type="button" id={`${base}-${id}`} aria-selected={tab === id} aria-controls={`${base}-${id}-panel`}
              tabIndex={tab === id ? 0 : -1} className={`tab${tab === id ? ' tab--active' : ''}`} onClick={() => setTab(id)}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="preview-actions">
        <button type="button" className="btn btn--secondary" onClick={runPreview} disabled={!file || preview.phase === 'loading'}>
          <Eye size={16} aria-hidden="true" /> {preview.phase === 'ready' ? 'Refresh preview' : 'Preview result'}
        </button>
        {!hasChanges && <span className="muted">No crop or filters yet; the preview equals the original.</span>}
      </div>

      <div role="tabpanel" id={`${base}-crop-panel`} aria-labelledby={`${base}-crop`} hidden={tab !== 'crop'}>
        <CropEditor src={previewUrl} dimensions={dimensions} crop={prepare.crop} onChange={prepare.setCrop} />
      </div>

      <div role="tabpanel" id={`${base}-preview-panel`} aria-labelledby={`${base}-preview`} hidden={tab !== 'preview'} aria-busy={preview.phase === 'loading'}>
        {preview.phase === 'error' && <p className="inline-note" role="alert">{preview.error.message}</p>}
        <div className="compare-pair">
          <figure>
            <figcaption className="card__title">Original</figcaption>
            <img src={previewUrl} alt="Original image" />
          </figure>
          <figure>
            <figcaption className="card__title">
              Processed {preview.info && <span className="muted">{preview.info.width} x {preview.info.height} px</span>}
            </figcaption>
            {preview.phase === 'loading' && <Skeleton width="100%" aspect={aspect} />}
            {preview.phase === 'ready' && <img src={preview.image} alt="Processed image preview" />}
            {(preview.phase === 'idle' || preview.phase === 'error') && (
              <div className="placeholder" style={{ aspectRatio: aspect }}>Select Preview result to see the processed image.</div>
            )}
            {preview.phase === 'ready' && preview.info.preprocessing.steps.length > 0 && (
              <ul className="engine-list" aria-label="Engine used per step">
                {preview.info.preprocessing.steps.map((s, i) => (
                  <li key={i}><span>{i + 1}. {s.label}</span><EngineBadge engine={s.engine} fallback={s.fallback} /><span className="muted">{Math.round(s.duration_ms)} ms</span></li>
                ))}
              </ul>
            )}
          </figure>
        </div>
      </div>
    </section>
  )
}
