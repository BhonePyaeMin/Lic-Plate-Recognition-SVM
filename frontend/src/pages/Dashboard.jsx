import { useEffect, useMemo, useRef, useState } from 'react'
import Header from '../components/Header.jsx'
import UploadPanel from '../features/detection/UploadPanel.jsx'
import ResultsPanel from '../features/detection/ResultsPanel.jsx'
import DetectBar from '../features/detection/DetectBar.jsx'
import FilterPanel from '../features/prepare/FilterPanel.jsx'
import ImageStage from '../features/prepare/ImageStage.jsx'
import ComparePanel from '../features/prepare/ComparePanel.jsx'
import LivePreview from '../features/prepare/LivePreview.jsx'
import { useDetection } from '../hooks/useDetection'
import { usePrepare } from '../hooks/usePrepare'
import { DEFAULT_LIMITS, getHealth } from '../services/api'

const STEPS = [
  { title: 'Upload', text: 'Choose a vehicle photo.' },
  { title: 'Prepare', text: 'Crop and add filters (optional).' },
  { title: 'Detect', text: 'Locate the plate, verify it with the SVM and read it.' },
]

export default function Dashboard() {
  const [health, setHealth] = useState({ state: 'checking', data: null })

  useEffect(() => {
    const controller = new AbortController()
    getHealth(controller.signal).then((data) => {
      if (!controller.signal.aborted) setHealth({ state: data ? 'online' : 'offline', data })
    })
    return () => controller.abort()
  }, [])

  const limits = useMemo(
    () =>
      health.data
        ? { maxUploadMb: health.data.max_upload_mb, allowedExtensions: health.data.allowed_extensions }
        : DEFAULT_LIMITS,
    [health.data],
  )

  const optionsRef = useRef({})
  const detection = useDetection(limits, () => optionsRef.current)
  const prepare = usePrepare(detection.file, health.state)
  optionsRef.current = { ...prepare.options, detectorMode: prepare.detectorMode, ocrEngine: prepare.ocrEngine }
  const hasFile = Boolean(detection.file)
  const busy = detection.phase === 'loading'

  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <Header health={health} />
      <main id="main" className="page">
        <section className={`hero${hasFile ? ' hero--compact' : ''}`} aria-labelledby="intro-title">
          <div className="hero__copy">
            <p className="hero__eyebrow">SVM-assisted plate recognition</p>
            <h1 id="intro-title">Detect and read vehicle plates, verified by an SVM</h1>
            <p className="hero__lead">Upload a photo, tune it with filters while a live preview updates beside the controls, then run detection.</p>
          </div>
          <ol className="steps">
            {STEPS.map((s, i) => (
              <li key={s.title}>
                <span className="steps__num" aria-hidden="true">{i + 1}</span>
                <span><strong>{s.title}.</strong> {s.text}</span>
              </li>
            ))}
          </ol>
        </section>
        <div className="studio">
          <aside className="rail" aria-label="Image and filter controls">
            {hasFile && <LivePreview detection={detection} prepare={prepare} />}
            <UploadPanel detection={detection} limits={limits} apiOffline={health.state === 'offline'} />
            {hasFile && (
              <section className="card" aria-labelledby="filters-title">
                <h2 id="filters-title" className="card__title card__title--spaced">Filters</h2>
                <FilterPanel prepare={prepare} mat={health.data?.matlab} svm={health.data?.svm} svmOcr={health.data?.svm_ocr} />
              </section>
            )}
            {hasFile && <DetectBar detection={detection} prepare={prepare} />}
          </aside>
          <div className="stage-col col">
            {hasFile && <ImageStage detection={detection} prepare={prepare} />}
            {hasFile && <ComparePanel prepare={prepare} disabled={busy || !prepare.catalog.operations.length} />}
            <ResultsPanel detection={detection} prepare={prepare} />
          </div>
        </div>
      </main>
    </>
  )
}
