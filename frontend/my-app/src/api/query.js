// src/api/query.js
import client from './client.js'
import { SAMPLE_QUESTIONS } from '../utils/constants.js'

/**
 * Submit a query to GraphRAG and/or VectorRAG.
 *
 * @param {string} query - The question to ask
 * @param {string} mode  - 'graphrag' | 'vectorrag' | 'both'
 * @param {object} options
 * @param {string} options.communityLevel - 'c0' | 'c1' | 'c2' | 'c3'
 * @param {boolean} options.includeContext - include source chunks/summaries
 * @param {number} options.maxContextTokens
 * @returns {Promise<QueryResponse>}
 */
export async function submitQuery(query, mode = 'both', options = {}) {
  const {
    communityLevel   = 'c1',
    includeContext   = true,
    maxContextTokens = 8000,
  } = options

  const response = await client.post('/query', {
    query,
    mode,
    community_level:    communityLevel,
    include_context:    includeContext,
    max_context_tokens: maxContextTokens,
  })
  return response.data
}

/**
 * Returns the hardcoded list of sample global sensemaking questions.
 * No API call — just pulls from constants so the page loads instantly.
 */
export function fetchSampleQuestions() {
  return SAMPLE_QUESTIONS
}

/**
 * Check if the backend is healthy and the index is ready.
 * @returns {Promise<{ healthy: boolean, indexed: boolean }>}
 */
export async function checkHealth() {
  try {
    const [health, ready] = await Promise.allSettled([
      client.get('/health'),
      client.get('/health/ready'),
    ])

    const healthy = health.status === 'fulfilled'
    const indexed = ready.status === 'fulfilled'
      && ready.value.data?.indexed === true

    return { healthy, indexed }
  } catch {
    return { healthy: false, indexed: false }
  }
}