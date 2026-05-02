// src/hooks/useGraphData.js
import { useCallback } from 'react'
import { useGraphContext } from '../context/GraphContext.jsx'
import { fetchGraphData, fetchCommunities } from '../api/graph.js'

/**
 * useGraphData — fetch and cache graph nodes, edges, and communities.
 *
 * Caches by level so switching between C0/C1/C2/C3 doesn't re-fetch.
 * Call fetchGraph(level) to load a level on demand.
 */
export function useGraphData() {
  const ctx = useGraphContext()

  const fetchGraph = useCallback(
    async (level = ctx.selectedLevel) => {
      // Skip if already cached
      if (ctx.isLevelCached(level)) {
        ctx.setLevel(level)
        return
      }

      ctx.dispatch({ type: 'FETCH_START' })
      ctx.setLevel(level)

      try {
        const [graphData, communities] = await Promise.allSettled([
          fetchGraphData(level),
          fetchCommunities(level),
        ])

        if (graphData.status === 'fulfilled') {
          ctx.dispatch({
            type: 'FETCH_SUCCESS',
            payload: {
              level,
              nodes: graphData.value.nodes,
              edges: graphData.value.edges,
              stats: graphData.value.stats,
            },
          })
        } else {
          throw graphData.reason
        }

        if (communities.status === 'fulfilled') {
          ctx.dispatch({
            type: 'COMMUNITIES_SUCCESS',
            payload: { level, communities: communities.value },
          })
        }
      } catch (err) {
        ctx.dispatch({
          type: 'FETCH_ERROR',
          payload: err.message || 'Failed to load graph data',
        })
      }
    },
    [ctx]
  )

  return {
    // Actions
    fetchGraph,
    selectNode:      ctx.selectNode,
    selectCommunity: ctx.selectCommunity,
    setEntityFilter: ctx.setEntityFilter,
    clearSelection:  ctx.clearSelection,
    setLevel:        ctx.setLevel,
    // State
    selectedLevel:      ctx.selectedLevel,
    currentGraphData:   ctx.currentGraphData,
    currentCommunities: ctx.currentCommunities,
    selectedNode:       ctx.selectedNode,
    selectedCommunity:  ctx.selectedCommunity,
    entityTypeFilter:   ctx.entityTypeFilter,
    isLoading:          ctx.isLoading,
    error:              ctx.error,
    isLevelCached:      ctx.isLevelCached,
  }
}