import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, Plus, Trash2, Wand2 } from 'lucide-react'
import { PRESETS } from '../../hooks/usePrepare'

const MODULES = { 1: 'Module 1: Enhancement', 2: 'Module 2: Segmentation and edges', 3: 'Module 3: Geometry' }

function ParamField({ param, value, onChange, id }) {
  if (param.kind === 'select') {
    return (
      <label className="field" htmlFor={id}>
        <span>{param.label}</span>
        <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
          {param.options.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      </label>
    )
  }
  const set = (raw) => {
    const n = Number(raw)
    if (raw !== '' && Number.isFinite(n)) onChange(n)
  }
  return (
    <div className="field field--range">
      <label htmlFor={id}>{param.label}</label>
      <div className="range-row">
        <input type="range" aria-label={`${param.label} slider`} min={param.min} max={param.max} step={param.step} value={value} onChange={(e) => set(e.target.value)} />
        <input id={id} type="number" min={param.min} max={param.max} step={param.step} value={value} onChange={(e) => set(e.target.value)} />
      </div>
    </div>
  )
}

export default function FilterPanel({ prepare, mat, svm, svmOcr }) {
  const { catalog, byId, steps, engine, detectorMode, ocrEngine, reference, needsReference } = prepare
  const [choice, setChoice] = useState('')

  const grouped = useMemo(() => {
    const out = {}
    for (const op of catalog.operations) {
      ;((out[op.module] ??= {})[op.group] ??= []).push(op)
    }
    return out
  }, [catalog.operations])

  const matlabNote = {
    ready: 'MATLAB is ready.',
    starting: 'MATLAB is starting; OpenCV is used until it is ready.',
    no_package: 'MATLAB Engine is not installed on the server, so OpenCV runs everything.',
    disabled: 'MATLAB is turned off on the server.',
    failed: 'MATLAB failed to start; OpenCV is used.',
  }[mat?.state] ?? 'MATLAB status unknown.'

  const svmNote = {
    ready: 'A trained SVM is loaded. It was trained on synthetic scenes only, so its accuracy on real photos is unknown.',
    no_model: 'No SVM model is installed on the server, so only the heuristic detector runs.',
    disabled: 'The SVM is turned off on the server.',
    failed: 'The SVM model could not be loaded; the heuristic detector is used.',
  }[svm?.state] ?? 'SVM status unknown.'

  const ocrNote = {
    ready: 'The SVM segments the plate into characters and classifies each one. Trained on synthetic plates only; in Automatic mode RapidOCR steps in when the SVM cannot read the plate.',
    no_model: 'No SVM character model is installed on the server, so RapidOCR reads the plate.',
    disabled: 'SVM character recognition is turned off on the server.',
    failed: 'The SVM character model could not be loaded; RapidOCR is used.',
  }[svmOcr?.state] ?? 'SVM character recognition status unknown.'

  return (
    <div className="filters">
      <details className="options" open>
      <summary>Detection and engine options</summary>
      <div className="engine">
        <label htmlFor="detector-select">Plate detection method</label>
        <select id="detector-select" value={detectorMode} onChange={(e) => prepare.setDetectorMode(e.target.value)}>
          <option value="auto">Automatic (SVM ranks regions when available)</option>
          <option value="contour">Heuristic only (OpenCV)</option>
          <option value="svm" disabled={!svm?.available}>SVM verified (rejects non-plate regions)</option>
        </select>
        <p className="muted">{svmNote}</p>
      </div>

      <div className="engine">
        <label htmlFor="ocr-select">Text recognition</label>
        <select id="ocr-select" value={ocrEngine} onChange={(e) => prepare.setOcrEngine(e.target.value)}>
          <option value="auto">Automatic (SVM first, RapidOCR if it cannot read)</option>
          <option value="svm" disabled={!svmOcr?.available}>SVM characters only (HOG + SVM)</option>
          <option value="rapidocr">RapidOCR only</option>
        </select>
        <p className="muted">{ocrNote}</p>
      </div>

      <div className="engine">
        <label htmlFor="engine-select">Processing engine</label>
        <select id="engine-select" value={engine} onChange={(e) => prepare.setEngine(e.target.value)}>
          <option value="auto">Automatic (MATLAB when ready, else OpenCV)</option>
          <option value="opencv">OpenCV only</option>
          <option value="matlab">MATLAB only</option>
        </select>
        <p className="muted">{matlabNote}</p>
      </div>
      </details>

      {catalog.state === 'unavailable' && (
        <p className="inline-note" role="note">Filters could not be loaded. Start the service and reload the page.</p>
      )}

      <div className="presets" role="group" aria-label="Quick presets">
        <span className="muted">Presets</span>
        {PRESETS.map((p) => (
          <button key={p.label} type="button" className="chip-btn" onClick={() => prepare.applyPreset(p)} disabled={!catalog.operations.length}>
            <Wand2 size={14} aria-hidden="true" /> {p.label}
          </button>
        ))}
      </div>

      <div className="add-filter">
        <label htmlFor="add-filter-select" className="sr-only">Choose a filter to add</label>
        <select id="add-filter-select" value={choice} onChange={(e) => setChoice(e.target.value)} disabled={!catalog.operations.length}>
          <option value="">Choose a filter...</option>
          {Object.entries(grouped).map(([mod, groups]) =>
            Object.entries(groups).map(([group, ops]) => (
              <optgroup key={`${mod}-${group}`} label={`${MODULES[mod]} - ${group}`}>
                {ops.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
              </optgroup>
            )),
          )}
        </select>
        <button type="button" className="btn btn--secondary" disabled={!choice || steps.length >= catalog.maxSteps}
          onClick={() => { prepare.addStep(choice); setChoice('') }}>
          <Plus size={16} aria-hidden="true" /> Add
        </button>
      </div>

      {steps.length === 0 ? (
        <p className="muted">No filters added. Detection will use the image as is.</p>
      ) : (
        <ol className="step-list" aria-label="Filter steps, applied in order">
          {steps.map((s, i) => {
            const op = byId[s.id]
            if (!op) return null
            return (
              <li key={s.uid} className="step">
                <div className="step__head">
                  <span className="step__num" aria-hidden="true">{i + 1}</span>
                  <div className="step__title">
                    <strong>{op.label}</strong>
                    <span className="muted">{op.description}</span>
                  </div>
                  <div className="step__tools">
                    <button type="button" className="icon-btn" aria-label={`Move ${op.label} up`} disabled={i === 0} onClick={() => prepare.moveStep(s.uid, -1)}><ArrowUp size={16} aria-hidden="true" /></button>
                    <button type="button" className="icon-btn" aria-label={`Move ${op.label} down`} disabled={i === steps.length - 1} onClick={() => prepare.moveStep(s.uid, 1)}><ArrowDown size={16} aria-hidden="true" /></button>
                    <button type="button" className="icon-btn" aria-label={`Remove ${op.label}`} onClick={() => prepare.removeStep(s.uid)}><Trash2 size={16} aria-hidden="true" /></button>
                  </div>
                </div>
                {op.output === 'mask' && (
                  <p className="step__warn">Produces a black and white mask. Great for comparing methods; text reading usually works better on a normal image.</p>
                )}
                {op.params.length > 0 && (
                  <div className="step__params">
                    {op.params.map((p) => (
                      <ParamField key={p.key} id={`${s.uid}-${p.key}`} param={p} value={s.params[p.key]} onChange={(v) => prepare.setParam(s.uid, p.key, v)} />
                    ))}
                  </div>
                )}
              </li>
            )
          })}
        </ol>
      )}

      {needsReference && (
        <div className="reference">
          <label htmlFor="reference-file">Reference image (needed by a selected filter)</label>
          <input id="reference-file" type="file" accept=".jpg,.jpeg,.png,.webp" onChange={(e) => prepare.setReference(e.target.files?.[0] ?? null)} />
          {reference ? <span className="muted">Using {reference.name}</span> : <span className="step__warn">Choose a reference image before previewing or detecting.</span>}
        </div>
      )}

      {steps.length > 0 && (
        <button type="button" className="btn btn--ghost btn--sm" onClick={prepare.clearSteps}>Remove all filters</button>
      )}
    </div>
  )
}
