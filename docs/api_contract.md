# API Contract

Complete reference for the GraphRAG vs VectorRAG REST API.

**Base URL:** `http://localhost:8000/api/v1`  
**Interactive docs:** `http://localhost:8000/docs` (Swagger UI)  
**Auth:** All endpoints except `/health` require `X-API-Key: <your-api-key>` header.

---

## Authentication

Every request (except health checks) must include:

```
X-API-Key: your-api-key-value
```

The API key is set via the `API_KEY` environment variable. A mismatch returns `401 Unauthorized`.

---

## Health

### `GET /health`
Liveness probe. Returns 200 if the process is running.

**Response 200**
```json
{ "status": "ok", "version": "1.0.0" }
```

### `GET /health/ready`
Readiness probe. Returns 200 only when the index is loaded and queries can be served.

**Response 200** — ready
```json
{ "status": "ready", "indexed": true }
```
**Response 503** — not ready (index not loaded)
```json
{ "status": "not_ready", "indexed": false, "message": "Index not yet available" }
```

---

## Query

### `POST /query`

Submit a question to GraphRAG, VectorRAG, or both systems simultaneously.

**Request body**
```json
{
  "query": "What are the major themes across the corpus?",
  "mode": "both",
  "community_level": "c1",
  "include_context": true,
  "max_context_tokens": 8000
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | The question to answer |
| `mode` | `graphrag` \| `vectorrag` \| `both` | `both` | Which system(s) to query |
| `community_level` | `c0` \| `c1` \| `c2` \| `c3` | `c1` | Community hierarchy level (GraphRAG only). c0=broadest, c3=most granular |
| `include_context` | boolean | `true` | Include source summaries/chunks in the response |
| `max_context_tokens` | integer | `8000` | Token limit for context window |

**Response 200**
```json
{
  "query": "What are the major themes across the corpus?",
  "mode": "both",
  "graphrag": {
    "answer": "The corpus covers three major themes...",
    "community_level": "c1",
    "communities_used": 12,
    "context": [...],
    "prompt_tokens": 4200,
    "completion_tokens": 380,
    "total_tokens": 4580,
    "latency_ms": 3240.5
  },
  "vectorrag": {
    "answer": "Based on the retrieved documents...",
    "chunks_retrieved": 8,
    "context": [...],
    "prompt_tokens": 2100,
    "completion_tokens": 290,
    "total_tokens": 2390,
    "latency_ms": 1820.1
  },
  "latency_ms": 3340.2
}
```

**Response 503** — corpus not indexed
```json
{ "detail": "Corpus not indexed. Run POST /index first." }
```

---

## Indexing

### `POST /index`

Trigger the full offline indexing pipeline. Runs asynchronously — returns a job ID immediately.

**Request body**
```json
{
  "chunk_size": 600,
  "chunk_overlap": 100,
  "gleaning_rounds": 2,
  "context_window_size": 8000,
  "max_community_levels": 3,
  "force_reindex": false,
  "skip_claims": false,
  "max_chunks": null
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `chunk_size` | integer | `600` | Tokens per chunk (paper uses 600) |
| `chunk_overlap` | integer | `100` | Overlap tokens between chunks |
| `gleaning_rounds` | integer | `2` | Self-reflection iterations per chunk |
| `context_window_size` | integer | `8000` | LLM context window for summarization |
| `max_community_levels` | integer | `3` | Max hierarchy depth (produces C0..CN) |
| `force_reindex` | boolean | `false` | Delete existing artifacts and restart |
| `skip_claims` | boolean | `false` | Skip claim extraction |
| `max_chunks` | integer \| null | `null` | Limit chunks for dev/testing |

**Response 202 Accepted**
```json
{
  "job_id": "idx_20240315_143022_abc123",
  "status": "queued",
  "message": "Indexing pipeline started. Poll /index/status for progress."
}
```

**Response 409 Conflict** — job already running
```json
{ "detail": "An indexing job is already running. Check /index/status." }
```

---

### `GET /index/status`

Poll the status of the most recent indexing job.

**Response 200**
```json
{
  "job_id": "idx_20240315_143022_abc123",
  "status": "running",
  "current_stage": "extraction",
  "stages": [
    { "name": "chunking",   "status": "completed", "progress": 100.0, "elapsed_s": 4.2 },
    { "name": "extraction", "status": "running",   "progress": 42.0,  "elapsed_s": 183.1 },
    { "name": "graph",      "status": "queued",    "progress": 0.0,   "elapsed_s": null },
    { "name": "communities","status": "queued",    "progress": 0.0,   "elapsed_s": null },
    { "name": "summarization","status":"queued",   "progress": 0.0,   "elapsed_s": null },
    { "name": "embeddings", "status": "queued",    "progress": 0.0,   "elapsed_s": null }
  ],
  "artifact_counts": {
    "chunks": 847,
    "entities": null,
    "relationships": null,
    "communities": null,
    "summaries": null
  },
  "started_at": "2024-03-15T14:30:22Z",
  "elapsed_seconds": 187.3
}
```

`status` values: `queued` | `running` | `completed` | `failed`

---

## Evaluation

### `POST /evaluate`

Run the LLM-as-judge evaluation suite. Submits each question to both systems and judges the responses on 4 criteria. Runs asynchronously.

**Request body**
```json
{
  "questions": [
    "What are the major themes?",
    "Who are the key organizations?"
  ],
  "criteria": ["comprehensiveness", "diversity", "empowerment", "directness"],
  "community_level": "c1",
  "runs_per_question": 5
}
```

| Criterion | Description |
|---|---|
| `comprehensiveness` | Does the answer cover all aspects of the question? |
| `diversity` | Does the answer present multiple perspectives and viewpoints? |
| `empowerment` | Does the answer help the reader understand and make decisions? |
| `directness` | Is the answer specific and well-organized? |

**Response 202 Accepted**
```json
{
  "eval_id": "eval_20240315_abc123",
  "status": "running",
  "questions": 2,
  "message": "Evaluation started. Poll /evaluation/results/{eval_id}."
}
```

---

### `GET /evaluation/results`

List all past evaluation runs.

**Response 200**
```json
[
  {
    "eval_id": "eval_20240315_abc123",
    "status": "completed",
    "questions": 8,
    "graphrag_win_rate": 0.625,
    "vectorrag_win_rate": 0.250,
    "tie_rate": 0.125,
    "created_at": "2024-03-15T16:00:00Z"
  }
]
```

---

### `GET /evaluation/results/{eval_id}`

Get full results for a specific evaluation run.

**Response 200**
```json
{
  "eval_id": "eval_20240315_abc123",
  "status": "completed",
  "summary": {
    "graphrag_win_rate": 0.625,
    "vectorrag_win_rate": 0.250,
    "tie_rate": 0.125,
    "by_criterion": {
      "comprehensiveness": { "graphrag": 0.75, "vectorrag": 0.125, "tie": 0.125 },
      "diversity":         { "graphrag": 0.75, "vectorrag": 0.125, "tie": 0.125 },
      "empowerment":       { "graphrag": 0.50, "vectorrag": 0.375, "tie": 0.125 },
      "directness":        { "graphrag": 0.50, "vectorrag": 0.375, "tie": 0.125 }
    }
  },
  "results": [
    {
      "question": "What are the major themes?",
      "graphrag_answer": "...",
      "vectorrag_answer": "...",
      "judgments": [...]
    }
  ]
}
```

---

## Graph

### `GET /graph`

Get statistics and metadata about the knowledge graph.

**Response 200**
```json
{
  "indexed": true,
  "node_count": 1247,
  "edge_count": 3891,
  "community_counts": { "c0": 4, "c1": 18, "c2": 67, "c3": 203 },
  "entity_types": { "PERSON": 312, "ORGANIZATION": 289, "CONCEPT": 646 },
  "chunk_count": 847
}
```

---

### `GET /communities/{level}`

List all communities at a given hierarchy level.

**Path parameter:** `level` — `c0` | `c1` | `c2` | `c3`

**Response 200**
```json
{
  "level": "c1",
  "communities": [
    {
      "community_id": "c1_007",
      "title": "AI Safety Research Community",
      "summary": "This community encompasses...",
      "impact_rating": 8.5,
      "node_count": 42,
      "key_entities": ["Anthropic", "OpenAI", "Paul Christiano"]
    }
  ],
  "total": 18
}
```

---

### `GET /communities/{level}/{community_id}`

Get the full summary for a specific community.

**Response 200** — full `CommunitySummary` object including all findings, node IDs, and context tokens used.

---

## Error Responses

All error responses follow this schema:

```json
{ "detail": "Human-readable error message" }
```

| Status | Meaning |
|---|---|
| `400` | Bad request — invalid parameters |
| `401` | Missing or invalid API key |
| `404` | Resource not found |
| `409` | Conflict — job already running |
| `422` | Validation error — request body schema violation |
| `429` | Rate limit exceeded |
| `503` | Service unavailable — corpus not indexed |