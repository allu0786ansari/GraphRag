// src/context/GraphContext.jsx
import { createContext, useContext, useReducer, useCallback } from 'react'

const GraphContext = createContext(null)

const initialState = {
  // Current community level being displayed
  selectedLevel: 'c1',
  // Node and edge data keyed by level — cache avoids re-fetching
  // { c0: { nodes, edges, stats }, c1: {...}, ... }
  graphDataByLevel: {},
  // Communities list keyed by level
  communitiesByLevel: {},
  // Currently selected node (for detail panel)
  selectedNode: null,
  // Currently highlighted community ID
  selectedCommunity: null,
  // Active entity type filter (null = show all)
  entityTypeFilter: null,
  // Loading state
  isLoading: false,
  // Error message
  error: null,
}

function graphReducer(state, action) {
  switch (action.type) {
    case 'SET_LEVEL':
      return { ...state, selectedLevel: action.payload, selectedNode: null, selectedCommunity: null }

    case 'FETCH_START':
      return { ...state, isLoading: true, error: null }

    case 'FETCH_SUCCESS': {
      const { level, nodes, edges, stats } = action.payload
      return {
        ...state,
        isLoading: false,
        graphDataByLevel: {
          ...state.graphDataByLevel,
          [level]: { nodes, edges, stats },
        },
      }
    }

    case 'COMMUNITIES_SUCCESS': {
      const { level, communities } = action.payload
      return {
        ...state,
        communitiesByLevel: {
          ...state.communitiesByLevel,
          [level]: communities,
        },
      }
    }

    case 'FETCH_ERROR':
      return { ...state, isLoading: false, error: action.payload }

    case 'SELECT_NODE':
      return { ...state, selectedNode: action.payload }

    case 'SELECT_COMMUNITY':
      return { ...state, selectedCommunity: action.payload }

    case 'SET_ENTITY_FILTER':
      return { ...state, entityTypeFilter: action.payload }

    case 'CLEAR_SELECTION':
      return { ...state, selectedNode: null, selectedCommunity: null }

    default:
      return state
  }
}

export function GraphProvider({ children }) {
  const [state, dispatch] = useReducer(graphReducer, initialState)

  const setLevel         = useCallback((l)  => dispatch({ type: 'SET_LEVEL', payload: l }), [])
  const selectNode       = useCallback((n)  => dispatch({ type: 'SELECT_NODE', payload: n }), [])
  const selectCommunity  = useCallback((c)  => dispatch({ type: 'SELECT_COMMUNITY', payload: c }), [])
  const setEntityFilter  = useCallback((f)  => dispatch({ type: 'SET_ENTITY_FILTER', payload: f }), [])
  const clearSelection   = useCallback(()   => dispatch({ type: 'CLEAR_SELECTION' }), [])

  // Derived: current level's graph data
  const currentGraphData   = state.graphDataByLevel[state.selectedLevel] || null
  const currentCommunities = state.communitiesByLevel[state.selectedLevel] || []
  const isLevelCached      = (level) => !!state.graphDataByLevel[level]

  const value = {
    ...state,
    currentGraphData,
    currentCommunities,
    isLevelCached,
    setLevel,
    selectNode,
    selectCommunity,
    setEntityFilter,
    clearSelection,
    dispatch,
  }

  return (
    <GraphContext.Provider value={value}>
      {children}
    </GraphContext.Provider>
  )
}

export function useGraphContext() {
  const ctx = useContext(GraphContext)
  if (!ctx) throw new Error('useGraphContext must be used inside <GraphProvider>')
  return ctx
}

export default GraphContext