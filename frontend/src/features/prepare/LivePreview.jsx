import { Eye } from 'lucide-react'

/**
 * Compact preview pinned to the top of the controls column. It always shows the
 * current result next to the sliders, so the full-size stage never has to be
 * scrolled into view while filters are being tuned.
 */
export default function LivePreview({ detection, prepare }) {
  const { previewUrl, dimensions } = detection
  const { preview } = prepare
  const changed = Boolean(prepare.crop) || prepare.steps.length > 0
  const loading = preview.phase === 'loading'
  const showProcessed = changed && Boolean(preview.image)
  const label = !changed ? 'Original' : loading ? 'Updating' : preview.phase === 'error' ? 'Preview failed' : 'Processed'
  const aspect = dimensions ? `${dimensions.width} / ${dimensions.height}` : '4 / 3'

  return (
    <section className="live" aria-label="Live preview">
      <div className="live__head">
        <span className="live__title"><Eye size={15} aria-hidden="true" /> Live preview</span>
        <span className={`live__state live__state--${loading ? 'busy' : preview.phase === 'error' ? 'error' : changed ? 'on' : 'off'}`} aria-live="polite">
          {label}
        </span>
      </div>
      <div className="live__frame" style={{ aspectRatio: aspect }}>
        <img src={showProcessed ? preview.image : previewUrl} alt="Live preview of the current image" className={loading ? 'is-updating' : ''} />
        {loading && <span className="live__bar" aria-hidden="true" />}
      </div>
    </section>
  )
}
