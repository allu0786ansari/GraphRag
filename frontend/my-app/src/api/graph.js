// src/api/graph.js
import client from './client.js'

/**
 * Fetch all nodes and edges for graph visualisation at a given community level.
 * @param {string} level - 'c0' | 'c1' | 'c2' | 'c3'
 * @returns {Promise<{ nodes: Node[], edges: Edge[], stats: object }>}
 */
export async function fetchGraphData(level = 'c1') {
  const response = await client.get('/graph', { params: { level } })
  const data = response.data

  // Normalise nodes and edges for react-force-graph-2d
  const nodes = (data.nodes || []).map((n) => ({
    id:          n.id || n.node_id,
    name:        n.name || n.label || n.id,
    type:        n.entity_type || n.type || 'DEFAULT',
    community:   n.community_id,
    degree:      n.degree || 1,
    description: n.description || '',
    ...n,
  }))

  const edges = (data.edges || []).map((e) => ({
    source: e.source || e.source_id,
    target: e.target || e.target_id,
    weight: e.weight || 1,
    label:  e.relationship_type || e.label || '',
    ...e,
  }))

  return { nodes, edges, stats: data.stats || {} }
}

/**
 * Fetch all communities at a given level.
 * @param {string} level - 'c0' | 'c1' | 'c2' | 'c3'
 * @returns {Promise<Community[]>}
 */
export async function fetchCommunities(level = 'c1') {
  const response = await client.get(`/communities/${level}`)
  return response.data?.communities || response.data || []
}

/**
 * Fetch the full detail for a specific community.
 * @param {string} level
 * @param {string} communityId
 * @returns {Promise<CommunityDetail>}
 */
export async function fetchCommunityDetail(level, communityId) {
  const response = await client.get(`/communities/${level}/${communityId}`)
  return response.data
}

/**
 * Fetch detail for a single graph node.
 * @param {string} nodeId
 * @returns {Promise<NodeDetail>}
 */
export async function fetchNodeDetail(nodeId) {
  const response = await client.get(`/graph/nodes/${encodeURIComponent(nodeId)}`)
  return response.data
}