import { useState } from 'react'
import { Cpu } from 'lucide-react'

const STATUS_LABEL = { readable: 'Readable', small: 'Small', unread: 'Not read' }

function engineLabel(name) {
  if (!name) return 'Unknown'
  if (name.startsWith('svm-hog-chars')) return 'SVM'
  if (name.startsWith('rapidocr')) return 'RapidOCR'
  return name
}

/**
 * Shows what the SVM did: every candidate plate on the analysed image (click to inspect),
 * the characters it segmented from the chosen plate, and the glyphs it classified.
 * Colours follow the legend: red = readable, orange = small, grey = not read.
 */
export default function SvmPanel({ result, imageUrl }) {
  const candidates = result.candidates ?? []
  const chosen = candidates.find((c) => c.selected) ?? candidates[0]
  const [pick, setPick] = useState(chosen?.index)
  if (!candidates.length) return null
  const current = candidates.find((c) => c.index === pick) ?? chosen
  const { image_width: W, image_height: H } = result
  const characters = current.characters

  return (
    <section className="card svm" aria-labelledby="svm-title">
      <div className="svm__head">
        <h3 id="svm-title" className="card__title"><Cpu size={16} aria-hidden="true" /> SVM recognition</h3>
        <ul className="svm__legend" aria-label="Legend">
          <li><span className="svm__swatch svm__swatch--readable" aria-hidden="true" />Readable</li>
          <li><span className="svm__swatch svm__swatch--small" aria-hidden="true" />Small</li>
          <li><span className="svm__swatch svm__swatch--unread" aria-hidden="true" />Not read</li>
        </ul>
      </div>
      <p className="muted">Plates are numbered on the image; the characters the SVM read are shown below. Click a plate to inspect it.</p>

      <div className="svm__grid">
        <div className="svm__frame" style={{ aspectRatio: `${W} / ${H}` }}>
          <img src={imageUrl} alt="Analyzed image with the candidate plates marked" />
          {candidates.map((c) => (
            <button
              key={c.index}
              type="button"
              className={`svm__box svm__box--${c.status}${c.index === current.index ? ' is-current' : ''}`}
              style={{ left: `${(c.box.x / W) * 100}%`, top: `${(c.box.y / H) * 100}%`, width: `${(c.box.width / W) * 100}%`, height: `${(c.box.height / H) * 100}%` }}
              aria-label={`Plate ${c.index}: ${c.text ?? 'not read'}`}
              aria-pressed={c.index === current.index}
              onClick={() => setPick(c.index)}
            >
              <span className="svm__tag">{c.index}{c.text ? `: ${c.text}` : ''}</span>
            </button>
          ))}
        </div>

        <div className="svm__detail" aria-live="polite">
          <p className="svm__caption">
            Plate {current.index}{current.text ? `: ${current.text}` : ''}
            <span className="muted">
              {' '}({characters != null ? `segmented, ${characters} character${characters === 1 ? '' : 's'}` : STATUS_LABEL[current.status].toLowerCase()})
            </span>
          </p>
          {current.segmented ? (
            <img className="svm__segmented" src={current.segmented} alt={`Plate ${current.index} with the characters found by the SVM boxed`} />
          ) : (
            <img className="svm__segmented" src={current.crop} alt={`Plate ${current.index} crop`} />
          )}
          {current.glyphs && (
            <figure className="svm__glyphs">
              <img src={current.glyphs} alt={`Plate ${current.index} characters as the SVM classified them`} />
              <figcaption className="muted">The characters exactly as the SVM received them, with its prediction below each.</figcaption>
            </figure>
          )}
          <div className="svm__chips">
            <span className={`chip chip--${current.status}`}>{STATUS_LABEL[current.status]}</span>
            {current.svm_score != null && (
              <span className="chip chip--muted" title="Margin of the plate-versus-background SVM. Positive means the plate side of its boundary; it is a different model from the character SVM and is not a probability.">
                Plate SVM score {current.svm_score.toFixed(2)}
              </span>
            )}
            <span className="chip chip--muted">Read by {engineLabel(current.ocr_engine)}</span>
          </div>
        </div>
      </div>

      {candidates.length > 1 && (
        <ul className="svm__list" aria-label="Candidate plates">
          {candidates.map((c) => (
            <li key={c.index}>
              <button
                type="button"
                className={`svm__thumb svm__thumb--${c.status}${c.index === current.index ? ' is-current' : ''}`}
                aria-pressed={c.index === current.index}
                onClick={() => setPick(c.index)}
              >
                <span className="svm__thumb-no">{c.index}</span>
                <img src={c.crop} alt="" />
                <span className="svm__thumb-text">{c.text ?? STATUS_LABEL[c.status]}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
