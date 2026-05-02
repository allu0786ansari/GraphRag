// src/api/evaluation.js
import client from './client.js'

/**
 * Trigger a new LLM-as-judge evaluation run.
 * Returns immediately — poll getEvalResults(evalId) for status.
 *
 * @param {object} options
 * @param {string[]} options.questions      - List of question strings to evaluate
 * @param {string[]} options.criteria       - ['comprehensiveness','diversity','empowerment','directness']
 * @param {string}   options.communityLevel - 'c0'|'c1'|'c2'|'c3'
 * @param {number}   options.runsPerQuestion - How many judge runs per Q (majority vote)
 * @returns {Promise<{ eval_id, status, questions, message }>}
 */
export async function runEvaluation(options = {}) {
  const {
    questions       = [],
    criteria        = ['comprehensiveness', 'diversity', 'empowerment', 'directness'],
    communityLevel  = 'c1',
    runsPerQuestion = 3,
  } = options

  const response = await client.post('/evaluate', {
    questions,
    criteria,
    community_level:   communityLevel,
    runs_per_question: runsPerQuestion,
  })
  return response.data
}

/**
 * List all past evaluation runs (summary only).
 * @returns {Promise<EvalSummary[]>}
 */
export async function listEvalResults() {
  const response = await client.get('/evaluation/results')
  return response.data || []
}

/**
 * Get full results for a specific evaluation run.
 * @param {string} evalId
 * @returns {Promise<EvalResult>}
 */
export async function getEvalResults(evalId) {
  const response = await client.get(`/evaluation/results/${evalId}`)
  return response.data
}