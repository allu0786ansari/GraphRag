// src/hooks/useEvaluation.js
import { useState, useCallback, useRef, useEffect } from 'react'
import { runEvaluation, listEvalResults, getEvalResults } from '../api/evaluation.js'

const POLL_INTERVAL_MS = 4000

/**
 * useEvaluation — trigger eval runs and subscribe to results.
 *
 * Manages its own local state (not in context) since evaluation
 * results don't need to be shared across pages.
 */
export function useEvaluation() {
  const [isRunning, setIsRunning]       = useState(false)
  const [currentEvalId, setCurrentEvalId] = useState(null)
  const [evalList, setEvalList]         = useState([])
  const [selectedResult, setSelectedResult] = useState(null)
  const [error, setError]               = useState(null)
  const [isLoadingList, setIsLoadingList] = useState(false)
  const [isLoadingResult, setIsLoadingResult] = useState(false)

  const pollTimerRef = useRef(null)
  const isMountedRef = useRef(true)

  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
      if (pollTimerRef.current) clearInterval(pollTimerRef.current)
    }
  }, [])

  // Load the list of past evaluation runs
  const loadEvalList = useCallback(async () => {
    setIsLoadingList(true)
    try {
      const list = await listEvalResults()
      if (isMountedRef.current) setEvalList(list)
    } catch (err) {
      if (isMountedRef.current) setError(err.message)
    } finally {
      if (isMountedRef.current) setIsLoadingList(false)
    }
  }, [])

  // Load a specific eval result
  const loadResult = useCallback(async (evalId) => {
    setIsLoadingResult(true)
    setSelectedResult(null)
    try {
      const result = await getEvalResults(evalId)
      if (isMountedRef.current) setSelectedResult(result)
    } catch (err) {
      if (isMountedRef.current) setError(err.message)
    } finally {
      if (isMountedRef.current) setIsLoadingResult(false)
    }
  }, [])

  // Poll for the current running eval
  const pollEval = useCallback(async (evalId) => {
    try {
      const result = await getEvalResults(evalId)
      if (!isMountedRef.current) return

      if (result.status === 'completed' || result.status === 'failed') {
        clearInterval(pollTimerRef.current)
        pollTimerRef.current = null
        setIsRunning(false)
        setSelectedResult(result)
        // Refresh the list
        await loadEvalList()
      }
    } catch {
      // Polling errors are non-fatal
    }
  }, [loadEvalList])

  // Trigger a new evaluation run
  const startEvaluation = useCallback(async (options = {}) => {
    setIsRunning(true)
    setError(null)
    setSelectedResult(null)

    try {
      const data = await runEvaluation(options)
      const evalId = data.eval_id
      setCurrentEvalId(evalId)

      // Start polling
      pollTimerRef.current = setInterval(() => pollEval(evalId), POLL_INTERVAL_MS)
    } catch (err) {
      if (isMountedRef.current) {
        setError(err.message || 'Failed to start evaluation')
        setIsRunning(false)
      }
    }
  }, [pollEval])

  return {
    // Actions
    startEvaluation,
    loadEvalList,
    loadResult,
    setSelectedResult,
    // State
    isRunning,
    currentEvalId,
    evalList,
    selectedResult,
    error,
    isLoadingList,
    isLoadingResult,
  }
}