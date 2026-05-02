// src/hooks/useIndexingStatus.js
import { useEffect, useRef, useCallback } from 'react'
import { useIndexingContext } from '../context/IndexingContext.jsx'
import { getIndexStatus, triggerIndexing, getGraphStats } from '../api/indexing.js'

const POLL_INTERVAL_MS = 3000

/**
 * useIndexingStatus — trigger indexing and poll pipeline status every 3s.
 *
 * Automatically stops polling when status becomes 'completed' or 'failed'.
 * On completion, fetches graph stats to populate the artifact stats panel.
 */
export function useIndexingStatus() {
  const ctx = useIndexingContext()
  const pollTimerRef = useRef(null)
  const isMountedRef = useRef(true)

  // Clean up on unmount
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
      if (pollTimerRef.current) clearInterval(pollTimerRef.current)
    }
  }, [])

  // Poll status
  const pollStatus = useCallback(async () => {
    try {
      const data = await getIndexStatus()
      if (!isMountedRef.current) return

      ctx.dispatch({ type: 'STATUS_UPDATE', payload: data })

      // Stop polling if terminal state
      if (data.status === 'completed' || data.status === 'failed') {
        if (pollTimerRef.current) {
          clearInterval(pollTimerRef.current)
          pollTimerRef.current = null
        }

        // On completion, load graph stats
        if (data.status === 'completed') {
          try {
            const stats = await getGraphStats()
            if (isMountedRef.current) {
              ctx.dispatch({ type: 'GRAPH_STATS_LOADED', payload: stats })
            }
          } catch {
            // Stats load failure is non-fatal
          }
        }
      }
    } catch (err) {
      if (isMountedRef.current) {
        ctx.dispatch({ type: 'STATUS_ERROR', payload: err.message })
      }
    }
  }, [ctx])

  // Start polling whenever isRunning becomes true
  useEffect(() => {
    if (ctx.isRunning && !pollTimerRef.current) {
      // Immediate first poll
      pollStatus()
      pollTimerRef.current = setInterval(pollStatus, POLL_INTERVAL_MS)
    }

    if (!ctx.isRunning && pollTimerRef.current) {
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [ctx.isRunning, pollStatus])

  // Trigger a new indexing job
  const startIndexing = useCallback(async (options = {}) => {
    try {
      const data = await triggerIndexing(options)
      ctx.dispatch({ type: 'JOB_STARTED', payload: data })
    } catch (err) {
      ctx.dispatch({
        type: 'JOB_ERROR',
        payload: err.message || 'Failed to start indexing',
      })
    }
  }, [ctx])

  // Load initial graph stats (call on page mount to check if already indexed)
  const loadGraphStats = useCallback(async () => {
    try {
      const stats = await getGraphStats()
      ctx.dispatch({ type: 'GRAPH_STATS_LOADED', payload: stats })
    } catch {
      // Not indexed yet — that's fine
    }
  }, [ctx])

  return {
    // Actions
    startIndexing,
    loadGraphStats,
    reset: ctx.reset,
    // State
    jobId:          ctx.jobId,
    status:         ctx.status,
    stages:         ctx.stages,
    currentStage:   ctx.currentStage,
    isRunning:      ctx.isRunning,
    artifactsReady: ctx.artifactsReady,
    graphStats:     ctx.graphStats,
    artifactCounts: ctx.artifactCounts,
    error:          ctx.error,
  }
}