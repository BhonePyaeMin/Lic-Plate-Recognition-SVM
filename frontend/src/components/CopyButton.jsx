import { useEffect, useRef, useState } from 'react'
import { Check, Copy } from 'lucide-react'

export default function CopyButton({ text, label = 'Copy plate text' }) {
  const [state, setState] = useState('idle') // idle | copied | failed
  const timer = useRef()
  useEffect(() => () => clearTimeout(timer.current), [])

  async function copy() {
    try {
      await navigator.clipboard.writeText(text)
      setState('copied')
    } catch {
      setState('failed')
    }
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setState('idle'), 2000)
  }

  const Icon = state === 'copied' ? Check : Copy
  const visible = state === 'copied' ? 'Copied' : state === 'failed' ? 'Copy failed' : 'Copy'
  return (
    <button type="button" className="btn btn--secondary" onClick={copy} aria-label={label}>
      <Icon size={16} aria-hidden="true" />
      <span aria-live="polite">{visible}</span>
    </button>
  )
}
