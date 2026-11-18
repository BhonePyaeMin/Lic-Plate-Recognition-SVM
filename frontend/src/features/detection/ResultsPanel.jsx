import { useEffect, useRef } from 'react'
import { ScanLine, SearchX, TriangleAlert } from 'lucide-react'
import ResultSkeleton from './ResultSkeleton.jsx'
import ResultView from './ResultView.jsx'

const ERROR_TITLES = {
  UNSUPPORTED_FILE_TYPE: 'Unsupported file type',
  FILE_TOO_LARGE: 'File too large',
  EMPTY_FILE: 'Empty file',
  INVALID_IMAGE: 'Image could not be read',
  API_UNAVAILABLE: 'Service unavailable',
  OCR_UNAVAILABLE: 'Text recognition unavailable',
  PROCESSING_FAILED: 'Processing failed',
  INVALID_PARAMS: 'Check the crop or filter settings',
  MATLAB_UNAVAILABLE: 'MATLAB unavailable',
}

function Notice({ icon: Icon, title, children, tone = 'neutral' }) {
  return (
    <div className={`notice notice--${tone}`} role={tone === 'danger' ? 'alert' : 'status'}>
      <span className="notice__icon" aria-hidden="true"><Icon size={22} /></span>
      <div>
        <h3 className="notice__title">{title}</h3>
        <p>{children}</p>
      </div>
    </div>
  )
}

export default function ResultsPanel({ detection, prepare }) {
  const { phase, data, error, previewUrl, dimensions } = detection
  const preprocessed = Boolean(prepare?.crop) || (prepare?.steps.length ?? 0) > 0
  const ref = useRef(null)

  // When a run starts or finishes, bring the results into view so nobody has to hunt for them.
  useEffect(() => {
    if (!['loading', 'success', 'no_plate', 'error'].includes(phase)) return
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    ref.current?.scrollIntoView?.({ behavior: reduce ? 'auto' : 'smooth', block: 'start' })
  }, [phase])

  return (
    <section ref={ref} className="results" aria-labelledby="results-title" aria-busy={phase === 'loading'}>
      <h2 id="results-title" className="sr-only">Results</h2>
      {phase === 'loading' && <ResultSkeleton dimensions={dimensions} showProcessed={preprocessed} />}
      {phase === 'success' && <ResultView data={data} previewUrl={previewUrl} />}
      {(phase === 'idle' || phase === 'ready') && (
        <div className="card empty">
          <Notice icon={ScanLine} title={phase === 'idle' ? 'No image yet' : 'Ready to analyze'}>
            {phase === 'idle'
              ? 'Upload a vehicle photo to see the detected plate, its crop, and the recognized text here.'
              : 'Crop and add filters if you like, then select Detect plate. Nothing is analyzed until you do.'}
          </Notice>
        </div>
      )}
      {phase === 'no_plate' && (
        <div className="card empty">
          <Notice icon={SearchX} title="No license plate detected">
            {data?.message ?? 'No plate was found.'} You can also try a different angle or better lighting.
          </Notice>
        </div>
      )}
      {phase === 'error' && (
        <div className="card empty">
          <Notice icon={TriangleAlert} tone="danger" title={ERROR_TITLES[error?.code] ?? 'Something went wrong'}>
            {error?.message}
          </Notice>
        </div>
      )}
    </section>
  )
}
