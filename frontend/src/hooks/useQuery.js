// src/hooks/useQuery.js
import { useCallback } from 'react'
import { useQueryContext } from '../context/QueryContext.jsx'
import { submitQuery as apiSubmitQuery } from '../api/query.js'

/**
 * useQuery — submit queries and manage query state.
 *
 * Returns:
 *   submitQuery(query, mode, options) — fires the API call
 *   All state from QueryContext (isLoading, error, graphragAnswer, etc.)
 */
export function useQuery() {
  const ctx = useQueryContext()

  const submitQuery = useCallback(
    async (query, mode, options = {}) => {
      if (!query?.trim()) return
      if (ctx.isLoading) return

      ctx.dispatch({ type: 'QUERY_START' })
      ctx.setQuery(query)

      try {
        const data = await apiSubmitQuery(query, mode || ctx.mode, {
          communityLevel: options.communityLevel || ctx.communityLevel,
          includeContext: options.includeContext ?? true,
          maxContextTokens: options.maxContextTokens ?? 8000,
        })
        ctx.dispatch({ type: 'QUERY_SUCCESS', payload: data })
      } catch (err) {
        ctx.dispatch({
          type: 'QUERY_ERROR',
          payload: err.message || 'Query failed — please try again',
        })
      }
    },
    [ctx]
  )

  return {
    // Actions
    submitQuery,
    setQuery:          ctx.setQuery,
    setMode:           ctx.setMode,
    setCommunityLevel: ctx.setCommunityLevel,
    clearQuery:        ctx.clearQuery,
    // State
    query:             ctx.query,
    mode:              ctx.mode,
    communityLevel:    ctx.communityLevel,
    isLoading:         ctx.isLoading,
    error:             ctx.error,
    graphragAnswer:    ctx.graphragAnswer,
    vectorragAnswer:   ctx.vectorragAnswer,
    rawResponse:       ctx.rawResponse,
    hasQueried:        ctx.hasQueried,
  }
}