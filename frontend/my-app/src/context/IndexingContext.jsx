// src/context/IndexingContext.jsx
import { createContext, useContext, useReducer, useCallback } from 'react'

const IndexingContext = createContext(null)

const initialState = {
  // Current job ID (null if no job has been triggered)
  jobId: null,
  // 'idle' | 'queued' | 'running' | 'completed' | 'failed'
  status: 'idle',
  // Array of stage objects from /index/status
  stages: [],
  // Which stage is currently active
  currentStage: null,
  // Whether a job is actively running (for polling)
  isRunning: false,
  // Whether the index is ready to serve queries
  artifactsReady: false,
  // Graph stats (node count, edge count, etc.) — populated after indexing
  graphStats: null,
  // Last status fetch error
  error: null,
  // Artifact counts (chunks, entities, etc.)
  artifactCounts: {},
}

function indexingReducer(state, action) {
  switch (action.type) {
    case 'JOB_STARTED':
      return {
        ...state,
        jobId: action.payload.job_id,
        status: 'queued',
        isRunning: true,
        error: null,
        stages: [],
        currentStage: null,
      }

    case 'STATUS_UPDATE': {
      const data = action.payload
      const isRunning = data.status === 'running' || data.status === 'queued'
      const completed = data.status === 'completed'
      return {
        ...state,
        status: data.status,
        stages: data.stages || state.stages,
        currentStage: data.current_stage || null,
        isRunning,
        artifactsReady: completed,
        artifactCounts: data.artifact_counts || state.artifactCounts,
        error: data.status === 'failed' ? (data.error || 'Pipeline failed') : null,
      }
    }

    case 'GRAPH_STATS_LOADED':
      return {
        ...state,
        graphStats: action.payload,
        artifactsReady: action.payload?.indexed === true,
      }

    case 'STATUS_ERROR':
      return { ...state, error: action.payload, isRunning: false }

    case 'JOB_ERROR':
      return { ...state, error: action.payload, isRunning: false, status: 'failed' }

    case 'RESET':
      return { ...initialState }

    default:
      return state
  }
}

export function IndexingProvider({ children }) {
  const [state, dispatch] = useReducer(indexingReducer, initialState)

  const reset = useCallback(() => dispatch({ type: 'RESET' }), [])

  const value = {
    ...state,
    reset,
    dispatch,
  }

  return (
    <IndexingContext.Provider value={value}>
      {children}
    </IndexingContext.Provider>
  )
}

export function useIndexingContext() {
  const ctx = useContext(IndexingContext)
  if (!ctx) throw new Error('useIndexingContext must be used inside <IndexingProvider>')
  return ctx
}

export default IndexingContext