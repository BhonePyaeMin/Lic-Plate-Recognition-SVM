import { useEffect, useId, useRef, useState } from 'react'
import { Eye } from 'lucide-react'
import CropEditor from './CropEditor.jsx'
import Skeleton from '../../components/Skeleton.jsx'
import EngineBadge from '../../components/EngineBadge.jsx'

/** Tabs: select the crop on the original, or preview the processed result next to the original. */
export default function ImageStage({ detection, prepare }) {
  const { previewUrl, dimensions, file } = detection
  const { preview } = prepare
  const [tab, setTab] = useState('crop')
  const [view, setView] = useState('fit') // 'fit' shrinks both images to the panel; 'actual' shows real pixels so a zoom is visible
  const base = useId()
  const hasChanges = Boolean(prepare.crop) || prepare.steps.length > 0

  // Adding a filter jumps to the preview so its effect is visible straight away.
  const stepCount = prepare.steps.length
  const prevCount = useRef(stepCount)
  useEffect(() => {
    if (stepCount > prevCount.current) setTab('preview')
    prevCount.current = stepCount
  }, [stepCount])

  async function runPreview() {
    setTab('preview')
    try { await prepare.run() } catch { /* shown via preview.error */ }
  }

  const aspect = dimensions ? `${dimensions.width} / ${dimensions.height}` : '4 / 3'
  const scaleRatio = dimensions && preview.info ? preview.info.width / dimensions.width : 1
  const ratio = Math.abs(scaleRatio - 1) > 0.01 ? scaleRatio.toFixed(2) : null

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
          <Eye size={16} aria-hidden="true" /> {preview.image ? 'Refresh preview' : 'Preview result'}
        </button>
        <span className="muted">{hasChanges ? 'Preview updates live as you change filters or the crop.' : 'No crop or filters yet; the preview equals the original.'}</span>
      </div>

      <div role="tabpanel" id={`${base}-crop-panel`} aria-labelledby={`${base}-crop`} hidden={tab !== 'crop'}>
        <CropEditor src={previewUrl} dimensions={dimensions} crop={prepare.crop} onChange={prepare.setCrop} />
      </div>

      <div role="tabpanel" id={`${base}-preview-panel`} aria-labelledby={`${base}-preview`} hidden={tab !== 'preview'} aria-busy={preview.phase === 'loading'}>
        {preview.phase === 'error' && <p className="inline-note" role="alert">{preview.error.message}</p>}
        <div className="view-switch" role="group" aria-label="Preview size">
          <button type="button" className={`view-switch__btn${view === 'fit' ? ' is-on' : ''}`} aria-pressed={view === 'fit'} onClick={() => setView('fit')}>Fit to panel</button>
          <button type="button" className={`view-switch__btn${view === 'actual' ? ' is-on' : ''}`} aria-pressed={view === 'actual'} onClick={() => setView('actual')}>Actual size (1:1)</button>
          {ratio && <span className="muted">Processed is {ratio}x the original size{view === 'fit' ? ' (both are shrunk to fit here; choose Actual size to see it)' : ''}.</span>}
        </div>
        <div className={`compare-pair${view === 'actual' ? ' compare-pair--actual' : ''}`} data-testid="compare-pair">
          <figure>
            <figcaption className="card__title">
              Original {dimensions && <span className="muted">{dimensions.width} x {dimensions.height} px</span>}
            </figcaption>
            <img src={previewUrl} alt="Original image" />
          </figure>
          <figure>
            <figcaption className="card__title">
              Processed {preview.info && <span className="muted">{preview.info.width} x {preview.info.height} px</span>}
            </figcaption>
            {preview.image ? (
              <div className={`live-img${preview.phase === 'loading' ? ' live-img--updating' : ''}`}>
                <img src={preview.image} alt="Processed image preview" />
                {preview.phase === 'loading' && <span className="live-img__badge" role="status">Updating preview</span>}
              </div>
            ) : preview.phase === 'loading' ? (
              <Skeleton width="100%" aspect={aspect} />
            ) : (
              <div className="placeholder" style={{ aspectRatio: aspect }}>Add a filter or draw a crop and the processed image appears here live.</div>
            )}
            {preview.info && preview.info.preprocessing.steps.length > 0 && (
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
