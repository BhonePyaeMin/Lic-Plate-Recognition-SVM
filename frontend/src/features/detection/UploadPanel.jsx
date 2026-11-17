import { useRef, useState } from 'react'
import { FileImage, Upload, X } from 'lucide-react'
import { formatBytes } from '../../utils/validateFile'

export default function UploadPanel({ detection, limits, apiOffline }) {
  const { file, previewUrl, phase, choose, clear } = detection
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const loading = phase === 'loading'
  const formats = limits.allowedExtensions.map((e) => e.toUpperCase()).join(', ')

  function onFiles(files) {
    if (files && files[0]) choose(files[0])
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    if (!loading) onFiles(e.dataTransfer.files)
  }

  return (
    <section className="card upload" aria-labelledby="upload-title">
      <h2 id="upload-title" className="card__title card__title--spaced">Vehicle image</h2>

      {!file ? (
        <div
          className={`dropzone${dragging ? ' dropzone--active' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <span className="dropzone__icon" aria-hidden="true"><Upload size={24} /></span>
          <p className="dropzone__lead">Drag an image here</p>
          <p className="muted">or</p>
          <button type="button" className="btn btn--secondary" onClick={() => inputRef.current?.click()}>
            Browse files
          </button>
          <p className="muted dropzone__hint">{formats}. Up to {limits.maxUploadMb} MB.</p>
        </div>
      ) : (
        <figure className="preview">
          <img src={previewUrl} alt="Selected vehicle" />
          <figcaption>
            <FileImage size={16} aria-hidden="true" />
            <span className="preview__name" title={file.name}>{file.name}</span>
            <span className="muted">{formatBytes(file.size)}</span>
            <button type="button" className="icon-btn" onClick={clear} disabled={loading} aria-label="Remove image">
              <X size={16} aria-hidden="true" />
            </button>
          </figcaption>
        </figure>
      )}

      <input
        ref={inputRef}
        className="sr-only"
        type="file"
        accept={limits.allowedExtensions.map((e) => `.${e}`).join(',')}
        aria-label="Choose a vehicle image"
        tabIndex={-1}
        onChange={(e) => { onFiles(e.target.files); e.target.value = '' }}
      />

      {apiOffline && (
        <p className="inline-note" role="note">
          The service is currently unreachable. You can still choose an image; detection will work once it is back.
        </p>
      )}

      {file && (
        <div className="upload__actions">
          <button type="button" className="btn btn--ghost btn--sm" onClick={() => inputRef.current?.click()} disabled={loading}>
            Replace image
          </button>
        </div>
      )}
    </section>
  )
}
