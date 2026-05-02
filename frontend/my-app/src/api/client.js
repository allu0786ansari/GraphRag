// src/api/client.js
// Central axios instance. Every API module imports from here.
// Handles: baseURL, auth header, timeout, error normalisation.

import axios from 'axios'
import { API_KEY } from '../utils/constants.js'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

const client = axios.create({
  // In dev: Vite proxies /api → backend:8000
  // In prod: nginx proxies /api → backend container
  baseURL: `${BASE_URL}/api/v1`,
  timeout: 120_000,   // 2 min — graphrag queries can be slow
  headers: {
    'Content-Type': 'application/json',
  },
})

// ── Request interceptor — attach API key ──────────────────────────────────────
client.interceptors.request.use(
  (config) => {
    const key = API_KEY || import.meta.env.VITE_API_KEY || ''
    if (key) {
      config.headers['X-API-Key'] = key
    }
    return config
  },
  (error) => Promise.reject(normaliseError(error))
)

// ── Response interceptor — normalise errors ───────────────────────────────────
client.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(normaliseError(error))
)

// ── Error normalisation ───────────────────────────────────────────────────────
// Every rejected promise from client carries: { message, status, code }
// Components only need to read error.message — no raw axios errors leak.

export function normaliseError(error) {
  if (axios.isAxiosError(error)) {
    const status  = error.response?.status
    const detail  = error.response?.data?.detail
    const message = detail
      || error.response?.data?.message
      || httpStatusMessage(status)
      || error.message
      || 'An unexpected error occurred'

    const normalised = new Error(message)
    normalised.status = status
    normalised.code   = error.code
    normalised.isApiError = true
    return normalised
  }
  return error
}

function httpStatusMessage(status) {
  switch (status) {
    case 400: return 'Bad request — check your input'
    case 401: return 'Unauthorised — check your API key in .env'
    case 403: return 'Forbidden'
    case 404: return 'Resource not found'
    case 409: return 'Conflict — a job may already be running'
    case 422: return 'Validation error — invalid request format'
    case 429: return 'Rate limit exceeded — please wait and retry'
    case 503: return 'Service unavailable — corpus may not be indexed yet'
    default:  return null
  }
}

export default client