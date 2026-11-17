import { ScanLine } from 'lucide-react'

function statusText(health) {
  if (health.state === 'checking') return 'Checking service'
  if (health.state === 'offline') return 'Service offline'
  const ocr = health.data?.ocr?.available ? 'OCR ready' : 'OCR unavailable'
  return `Service online, ${ocr}`
}

const MATLAB_LABEL = {
  ready: 'MATLAB: ready',
  starting: 'MATLAB: starting',
  no_package: 'MATLAB: not installed',
  disabled: 'MATLAB: off',
  failed: 'MATLAB: failed to start',
}

export default function Header({ health }) {
  const matlab = health.data?.matlab
  return (
    <header className="header">
      <div className="header__inner">
        <div className="brand">
          <span className="brand__icon" aria-hidden="true"><ScanLine size={20} /></span>
          <span className="brand__name">Plate Reader</span>
        </div>
        <div className="header__status" role="status">
          <span className={`dot dot--${health.state}`} aria-hidden="true" />
          <span>{statusText(health)}</span>
          {matlab && (
            <span className={`header__note header__note--${matlab.state}`} title={matlab.detail}>
              {MATLAB_LABEL[matlab.state] ?? 'MATLAB: unknown'}
            </span>
          )}
        </div>
      </div>
    </header>
  )
}
