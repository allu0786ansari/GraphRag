// src/api/indexing.js
import client from './client.js'

/**
 * Trigger the full indexing pipeline.
 * Returns immediately with a job_id — poll getIndexStatus() for progress.
 *
 * @param {object} options
 * @param {number}  options.chunkSize
 * @param {number}  options.chunkOverlap
 * @param {number}  options.gleaningRounds
 * @param {number}  options.contextWindowSize
 * @param {boolean} options.forceReindex
 * @param {boolean} options.skipClaims
 * @param {number|null} options.maxChunks  - null = full corpus
 * @returns {Promise<{ job_id, status, message }>}
 */
export async function triggerIndexing(options = {}) {
  const {
    chunkSize        = 600,
    chunkOverlap     = 100,
    gleaningRounds   = 0,
    contextWindowSize = 8000,
    forceReindex     = false,
    skipClaims       = false,
    maxChunks        = null,
  } = options

  const response = await client.post('/index', {
    chunk_size:          chunkSize,
    chunk_overlap:       chunkOverlap,
    gleaning_rounds:     gleaningRounds,
    context_window_size: contextWindowSize,
    force_reindex:       forceReindex,
    skip_claims:         skipClaims,
    max_chunks:          maxChunks,
  })
  return response.data
}

/**
 * Poll the status of the current (or most recent) indexing job.
 * @returns {Promise<IndexStatus>}
 */
export async function getIndexStatus() {
  const response = await client.get('/index/status')
  return response.data
}

/**
 * Get high-level statistics about the knowledge graph.
 * @returns {Promise<GraphStats>}
 */
export async function getGraphStats() {
  const response = await client.get('/graph')
  return response.data
}