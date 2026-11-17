import { useCallback, useEffect, useRef, useState } from 'react'
import { detectPlate } from '../services/api'
import { validateFile } from '../utils/validateFile'

/**
 * Phases: idle (no file) -> ready (valid file chosen) -> loading ->
 * success | no_plate | error. Processing only starts from `run()`.
 * `prepare` carries the user's crop, filter chain and engine choice.
 */
export function useDetection(limits, prepare) {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [dimensions, setDimensions] = useState(null)
  const [phase, setPhase] = useState('idle')
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const abortRef = useRef(null)
  const prepareRef = useRef(prepare)
  prepareRef.current = prepare // a function returning the latest crop/filter options

  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl])
  useEffect(() => () => abortRef.current?.abort(), [])

  const choose = useCallback(
    (candidate) => {
      abortRef.current?.abort()
      setData(null)
      const problem = validateFile(candidate, limits)
      if (problem) {
        setFile(null)
        setPreviewUrl(null)
        setDimensions(null)
        setError(problem)
        setPhase('error')
        return
      }
      const url = URL.createObjectURL(candidate)
      const probe = new Image()
      probe.onload = () => setDimensions({ width: probe.naturalWidth, height: probe.naturalHeight })
      probe.onerror = () => {
        setError({ code: 'INVALID_IMAGE', message: 'This file could not be read as an image. It may be corrupted; try another file.' })
        setFile(null)
        setPreviewUrl(null)
        setPhase('error')
      }
      probe.src = url
      setDimensions(null)
      setFile(candidate)
      setPreviewUrl(url)
      setError(null)
      setPhase('ready')
    },
    [limits],
  )

  const clear = useCallback(() => {
    abortRef.current?.abort()
    setFile(null)
    setPreviewUrl(null)
    setDimensions(null)
    setData(null)
    setError(null)
    setPhase('idle')
  }, [])

  const run = useCallback(async () => {
    if (!file) return
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setPhase('loading')
    setError(null)
    setData(null)
    try {
      const out = await detectPlate(file, prepareRef.current?.() ?? {}, controller.signal)
      if (controller.signal.aborted) return
      setData(out)
      setPhase(out.found ? 'success' : 'no_plate')
    } catch (err) {
      if (err.name === 'AbortError') return
      setError({ code: err.code ?? 'PROCESSING_FAILED', message: err.message })
      setPhase('error')
    }
  }, [file])

  return { file, previewUrl, dimensions, phase, data, error, choose, clear, run }
}
