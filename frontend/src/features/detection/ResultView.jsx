import { CircleCheck, Download, Info } from 'lucide-react'
import CopyButton from '../../components/CopyButton.jsx'
import EngineBadge from '../../components/EngineBadge.jsx'

function Figure({ title, src, alt, className = '', action }) {
  return (
    <figure className={`card figure ${className}`}>
      <figcaption className="figure__head">
        <span className="card__title">{title}</span>
        {action}
      </figcaption>
      <img src={src} alt={alt} />
    </figure>
  )
}

export default function ResultView({ data, previewUrl }) {
  const r = data.result
  const recognized = r.detection_status === 'recognized'
  const box = r.bounding_box
  const pre = r.preprocessing
  const steps = pre?.steps ?? []
  const rows = [
    ['Plate region', `${box.width} x ${box.height} px at (${box.x}, ${box.y})`],
    ['Image size', `${r.image_width} x ${r.image_height} px`],
    ['Processing time', `${Math.round(r.processing_time_ms)} ms`],
    ['Detector', r.detector],
    ['Region score', r.detector.startsWith('user-crop') ? 'Not applicable (the whole selected area was read)' : `${r.detector_score.toFixed(2)} (ranking heuristic, not a probability)`],
    ['OCR engine', r.ocr_engine ?? 'Not run'],
    ['Raw OCR output', r.raw_ocr_text ?? 'None'],
    ['MATLAB', r.matlab_used ? `Used in ${steps.filter((s) => s.engine === 'matlab').length} step(s)` : 'Not used'],
  ]
  if (pre?.crop) rows.splice(2, 0, ['Crop applied', `${pre.crop.width} x ${pre.crop.height} px at (${pre.crop.x}, ${pre.crop.y}) in the original`])

  return (
    <div className="result" data-testid="result-view">
      <p className="status-label status-label--done" role="status">
        <CircleCheck size={16} aria-hidden="true" />
        {recognized ? 'Analysis complete' : 'Region found, text unreadable'}
      </p>

      <div className="card plate-card">
        <span className="card__title">Recognized plate</span>
        {recognized ? (
          <div className="plate-card__row">
            <p className="plate-text" aria-label={`Plate text ${r.plate_text.split('').join(' ')}`}>{r.plate_text}</p>
            <CopyButton text={r.plate_text} />
          </div>
        ) : (
          <p className="plate-card__unreadable">
            <Info size={18} aria-hidden="true" />
            A possible plate region was found, but its text could not be read. Try a sharper or closer photo.
          </p>
        )}
        <div className="plate-card__meta">
          {recognized && r.confidence !== null && (
            <span className="chip">OCR confidence {Math.round(r.confidence * 100)}%</span>
          )}
          <span className="chip chip--muted">Verify the text against the image before relying on it</span>
        </div>
      </div>

      <div className="figures">
        <Figure title="Original" src={previewUrl} alt="Original uploaded image" />
        <Figure
          title="Detected plate"
          src={r.annotated_image}
          alt="Image with the detected plate outlined"
          action={
            <a className="btn btn--ghost btn--sm" href={r.annotated_image} download="annotated-plate.jpg">
              <Download size={16} aria-hidden="true" />
              <span>Download</span>
            </a>
          }
        />
      </div>

      {r.input_image && (
        <Figure title="Image analyzed (after crop and filters)" src={r.input_image} alt="Image after crop and filters, as seen by the detector" />
      )}

      <Figure title="Plate crop" className="figure--crop" src={r.plate_crop} alt="Cropped license plate region" />

      <section className="card details" aria-labelledby="details-title">
        <h3 id="details-title" className="card__title">Details</h3>
        <dl>
          {rows.map(([k, v]) => (
            <div className="details__row" key={k}>
              <dt>{k}</dt>
              <dd>{v}</dd>
            </div>
          ))}
          {steps.length > 0 && (
            <div className="details__row">
              <dt>Filters applied</dt>
              <dd>
                <ul className="engine-list">
                  {steps.map((s, i) => (
                    <li key={i}><span>{i + 1}. {s.label}</span><EngineBadge engine={s.engine} fallback={s.fallback} /></li>
                  ))}
                </ul>
              </dd>
            </div>
          )}
        </dl>
      </section>
    </div>
  )
}
