import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getFilters, previewFilters } from '../services/api'

let uid = 0
const nextUid = () => `s${++uid}`

export const PRESETS = [
  { label: 'Low light', steps: [['gamma', { gamma: 0.6 }], ['hist_equalization', {}]] },
  { label: 'Noisy photo', steps: [['median', { kernel: 3 }], ['laplacian', { strength: 0.6 }]] },
  { label: 'Soft focus', steps: [['highpass', { kernel: 5, boost: 1.8 }]] },
]

/** Crop, filter chain, engine, reference image and preview state for the "Prepare" stage. */
export function usePrepare(file, healthState) {
  const [catalog, setCatalog] = useState({ state: 'loading', operations: [], maxSteps: 8 })
  const [steps, setSteps] = useState([])
  const [crop, setCrop] = useState(null)
  const [engine, setEngine] = useState('auto')
  const [detectorMode, setDetectorMode] = useState('auto') // plate-region method; does not affect the filter preview
  const [ocrEngine, setOcrEngine] = useState('auto') // text recognition: auto = SVM first, RapidOCR if it cannot read; detect-only
  const [reference, setReference] = useState(null)
  const [preview, setPreview] = useState({ phase: 'idle', image: null, info: null, error: null })
  const abortRef = useRef(null)

  useEffect(() => {
    const controller = new AbortController()
    getFilters(controller.signal).then((data) => {
      if (controller.signal.aborted) return
      setCatalog(
        data
          ? { state: 'ready', operations: data.operations, maxSteps: data.max_steps }
          : { state: 'unavailable', operations: [], maxSteps: 8 },
      )
    })
    return () => controller.abort()
  }, [healthState])

  const byId = useMemo(() => Object.fromEntries(catalog.operations.map((o) => [o.id, o])), [catalog])

  // A new image invalidates the crop and any preview.
  useEffect(() => {
    setCrop(null)
    setPreview({ phase: 'idle', image: null, info: null, error: null })
  }, [file])

  // Changes no longer clear the preview: the live-preview effect below re-renders it.
  const invalidate = useCallback(() => {}, [])

  const makeStep = useCallback(
    (id, overrides = {}) => {
      const op = byId[id]
      if (!op) return null
      const params = Object.fromEntries(op.params.map((p) => [p.key, p.default]))
      return { uid: nextUid(), id, params: { ...params, ...overrides } }
    },
    [byId],
  )

  const addStep = useCallback(
    (id) => {
      const step = makeStep(id)
      if (!step) return
      setSteps((s) => (s.length >= catalog.maxSteps ? s : [...s, step]))
      invalidate()
    },
    [makeStep, catalog.maxSteps, invalidate],
  )

  const applyPreset = useCallback(
    (preset) => {
      setSteps(preset.steps.map(([id, o]) => makeStep(id, o)).filter(Boolean))
      invalidate()
    },
    [makeStep, invalidate],
  )

  const removeStep = useCallback((stepUid) => { setSteps((s) => s.filter((x) => x.uid !== stepUid)); invalidate() }, [invalidate])
  const moveStep = useCallback((stepUid, dir) => {
    setSteps((s) => {
      const i = s.findIndex((x) => x.uid === stepUid)
      const j = i + dir
      if (i < 0 || j < 0 || j >= s.length) return s
      const copy = [...s]
      ;[copy[i], copy[j]] = [copy[j], copy[i]]
      return copy
    })
    invalidate()
  }, [invalidate])
  const setParam = useCallback((stepUid, key, value) => {
    setSteps((s) => s.map((x) => (x.uid === stepUid ? { ...x, params: { ...x.params, [key]: value } } : x)))
    invalidate()
  }, [invalidate])
  const clearSteps = useCallback(() => { setSteps([]); invalidate() }, [invalidate])
  const updateCrop = useCallback((c) => { setCrop(c); invalidate() }, [invalidate])
  const updateEngine = useCallback((e) => { setEngine(e); invalidate() }, [invalidate])
  const updateReference = useCallback((f) => { setReference(f); invalidate() }, [invalidate])

  const needsReference = steps.some((s) => byId[s.id]?.needs_reference)
  const options = useMemo(() => ({ crop, steps, engine, reference }), [crop, steps, engine, reference])

  /** Run crop + chain (optionally plus extra steps) on the server and return the result. */
  const run = useCallback(
    async (extraSteps = [], { track = true } = {}) => {
      if (!file) return null
      // Only the tracked preview is cancellable; untracked runs (comparisons) go in parallel.
      const controller = new AbortController()
      if (track) {
        abortRef.current?.abort()
        abortRef.current = controller
      }
      if (track) setPreview((p) => ({ ...p, phase: 'loading', error: null })) // keep the previous image while updating
      try {
        const res = await previewFilters(file, { ...options, steps: [...steps, ...extraSteps] }, controller.signal)
        if (controller.signal.aborted) return null
        if (track) setPreview({ phase: 'ready', image: res.image, info: res, error: null })
        return res
      } catch (err) {
        if (err.name === 'AbortError') return null
        const error = { code: err.code ?? 'PROCESSING_FAILED', message: err.message }
        if (track) setPreview({ phase: 'error', image: null, info: null, error })
        throw error
      }
    },
    [file, options, steps],
  )

  // Live preview: re-run shortly after any change to the crop or filter chain.
  useEffect(() => {
    if (!file) return undefined
    if (!steps.length && !crop) {
      abortRef.current?.abort()
      setPreview({ phase: 'idle', image: null, info: null, error: null })
      return undefined
    }
    if (needsReference && !reference) return undefined
    const timer = setTimeout(() => { run().catch(() => {}) }, 350)
    return () => clearTimeout(timer)
  }, [run, file, steps, crop, needsReference, reference])

  return {
    catalog, byId, steps, crop, engine, detectorMode, ocrEngine, reference, preview, needsReference, options,
    addStep, applyPreset, removeStep, moveStep, setParam, clearSteps,
    setCrop: updateCrop, setEngine: updateEngine, setDetectorMode, setOcrEngine, setReference: updateReference, run,
  }
}
