// src/context/QueryContext.jsx
import { createContext, useContext, useReducer, useCallback } from 'react'

const QueryContext = createContext(null)

const initialState = {
  // Current query text
  query: '',
  // Query mode: 'both' | 'graphrag' | 'vectorrag'
  mode: 'both',
  // Community level for GraphRAG: 'c0' | 'c1' | 'c2' | 'c3'
  communityLevel: 'c1',
  // Loading state
  isLoading: false,
  // Error message string or null
  error: null,
  // GraphRAG answer object or null
  graphragAnswer: null,
  // VectorRAG answer object or null
  vectorragAnswer: null,
  // Full raw response from /query
  rawResponse: null,
  // Whether at least one successful query has been made
  hasQueried: false,
}

function queryReducer(state, action) {
  switch (action.type) {
    case 'SET_QUERY':
      return { ...state, query: action.payload }

    case 'SET_MODE':
      return { ...state, mode: action.payload }

    case 'SET_COMMUNITY_LEVEL':
      return { ...state, communityLevel: action.payload }

    case 'QUERY_START':
      return {
        ...state,
        isLoading: true,
        error: null,
        graphragAnswer: null,
        vectorragAnswer: null,
        rawResponse: null,
      }

    case 'QUERY_SUCCESS': {
      const data = action.payload
      return {
        ...state,
        isLoading: false,
        error: null,
        rawResponse: data,
        graphragAnswer: data.graphrag || null,
        vectorragAnswer: data.vectorrag || null,
        hasQueried: true,
      }
    }

    case 'QUERY_ERROR':
      return {
        ...state,
        isLoading: false,
        error: action.payload,
        graphragAnswer: null,
        vectorragAnswer: null,
      }

    case 'CLEAR':
      return {
        ...initialState,
        mode: state.mode,
        communityLevel: state.communityLevel,
      }

    default:
      return state
  }
}

export function QueryProvider({ children }) {
  const [state, dispatch] = useReducer(queryReducer, initialState)

  const setQuery         = useCallback((q)  => dispatch({ type: 'SET_QUERY', payload: q }), [])
  const setMode          = useCallback((m)  => dispatch({ type: 'SET_MODE', payload: m }), [])
  const setCommunityLevel = useCallback((l) => dispatch({ type: 'SET_COMMUNITY_LEVEL', payload: l }), [])
  const clearQuery       = useCallback(()   => dispatch({ type: 'CLEAR' }), [])

  const value = {
    ...state,
    setQuery,
    setMode,
    setCommunityLevel,
    clearQuery,
    dispatch,
  }

  return (
    <QueryContext.Provider value={value}>
      {children}
    </QueryContext.Provider>
  )
}

export function useQueryContext() {
  const ctx = useContext(QueryContext)
  if (!ctx) throw new Error('useQueryContext must be used inside <QueryProvider>')
  return ctx
}

export default QueryContext