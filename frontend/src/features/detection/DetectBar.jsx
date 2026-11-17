import { RotateCw, ScanSearch } from 'lucide-react'

export default function DetectBar({ detection, prepare }) {
  const { phase, run } = detection
  const loading = phase === 'loading'
  const retry = phase === 'error'
  const missingRef = prepare.needsReference && !prepare.reference
  const parts = []
  parts.push(prepare.crop ? `crop ${prepare.crop.width} x ${prepare.crop.height}` : 'full image')
  parts.push(prepare.steps.length ? `${prepare.steps.length} filter${prepare.steps.length > 1 ? 's' : ''}` : 'no filters')
  parts.push(`engine: ${prepare.engine}`)

  return (
    <section className="card detect-bar" aria-label="Run detection">
      <p className="muted" data-testid="detect-summary">Will analyze: {parts.join(', ')}.</p>
      <button type="button" className="btn btn--primary btn--block" onClick={run} disabled={loading || missingRef}>
        {retry ? <RotateCw size={18} aria-hidden="true" /> : <ScanSearch size={18} aria-hidden="true" />}
        {loading ? 'Analyzing image' : retry ? 'Try again' : 'Detect plate'}
      </button>
      {missingRef && <p className="step__warn">Choose the reference image first.</p>}
    </section>
  )
}
