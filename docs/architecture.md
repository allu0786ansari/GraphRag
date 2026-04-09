# Architecture

System design for the GraphRAG vs VectorRAG comparison platform.

---

## Overview

This system implements the paper *"From Local to Global: A Graph RAG Approach to Query-Focused Summarization"* (Edge et al., 2024) and provides a side-by-side comparison with a standard VectorRAG (dense retrieval) system.

The core thesis: **graph-structured community summaries enable better global sensemaking** over large document corpora than vector similarity search, particularly for queries that require synthesizing information from many disparate documents.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│              React SPA (Vite) — port 5173                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP / REST
┌─────────────────────────▼───────────────────────────────────────┐
│                     Nginx Reverse Proxy                         │
│         /api/v1/* → backend:8000  |  /* → frontend:80          │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│                     (uvicorn, port 8000)                        │
│                                                                  │
│  ┌─────────────┐  ┌───────────────┐  ┌────────────────────┐   │
│  │ Query API   │  │ Indexing API  │  │  Evaluation API    │   │
│  │ POST /query │  │ POST /index   │  │  POST /evaluate    │   │
│  └──────┬──────┘  └───────┬───────┘  └─────────┬──────────┘   │
│         │                 │                     │               │
│  ┌──────▼──────┐  ┌───────▼───────┐  ┌─────────▼──────────┐   │
│  │ GraphRAG    │  │   Pipeline    │  │    Evaluation      │   │
│  │ Engine      │  │   Runner      │  │    Engine          │   │
│  ├─────────────┤  │               │  └────────────────────┘   │
│  │ VectorRAG   │  │ Chunking      │                            │
│  │ Engine      │  │ Extraction    │                            │
│  └──────┬──────┘  │ Graph Build   │                            │
│         │         │ Communities   │                            │
│  ┌──────▼──────┐  │ Summarization │                            │
│  │  OpenAI     │  │ Embeddings    │                            │
│  │  Service    │  └───────┬───────┘                            │
│  └─────────────┘          │                                    │
│                           │                                    │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │                      Storage Layer                       │  │
│  │  ArtifactStore  │  GraphStore  │  SummaryStore           │  │
│  │  FAISSService   │  CacheManager                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────┐
│              Persistent Storage               │
│  data/processed/chunks.json                   │
│  data/processed/extractions.json              │
│  data/processed/graph.pkl                     │
│  data/processed/community_map.json            │
│  data/processed/community_summaries.json      │
│  data/processed/faiss_index.bin               │
│  data/processed/embeddings.npy                │
└───────────────────────────────────────────────┘
```

---

## Indexing Pipeline

The offline indexing pipeline transforms raw documents into the structured artifacts required for query-time retrieval. Runs once per corpus (or on corpus updates).

```
Raw Documents (.json)
        │
        ▼
┌───────────────┐
│  1. Chunking  │  Split docs into 600-token chunks with 100-token overlap
└───────┬───────┘  → chunks.json
        │
        ▼
┌───────────────┐
│ 2. Extraction │  LLM extracts entities, relationships, and claims per chunk
└───────┬───────┘  → extractions.json (incremental, resumable)
        │
        ▼
┌───────────────┐
│  3. Gleaning  │  2 self-reflection rounds to catch missed entities
└───────┬───────┘  (integrated into extraction stage)
        │
        ▼
┌───────────────┐
│ 4. Graph Build│  Deduplicate entities + merge into NetworkX graph
└───────┬───────┘  → graph.pkl
        │
        ▼
┌───────────────┐
│ 5. Communities│  Leiden algorithm: hierarchical community detection
└───────┬───────┘  → community_map.json (C0..C3 levels)
        │
        ▼
┌───────────────┐
│6. Summarization│  LLM summary per community (title, findings, impact)
└───────┬───────┘  → community_summaries.json
        │
        ▼
┌───────────────┐
│ 7. Embeddings │  Embed all chunks → FAISS index for VectorRAG
└───────────────┘  → faiss_index.bin + embeddings.npy
```

**Key design decisions:**
- **Resumable**: `CacheManager` tracks completed stages; re-running resumes from last checkpoint
- **Incremental**: Extractions are written per-chunk so a crash loses at most one chunk's work
- **Configurable**: All stage parameters (chunk_size, gleaning_rounds, etc.) are runtime config

---

## Query Engines

### GraphRAG Engine

Implements the map-reduce query pattern from the paper (Section 3):

```
User Query
    │
    ▼
Load community summaries (at specified level: C0–C3)
    │
    ▼
MAP stage: for each community summary in parallel
    │  ├── Build context from community summary + top entities
    │  ├── LLM: "Given this context, partially answer the query"
    │  └── Returns: {answer, score} — discard if score ≤ threshold
    │
    ▼
REDUCE stage: aggregate all partial answers
    │  ├── Sort by helpfulness score, take top-K
    │  ├── LLM: "Synthesize these partial answers into a final answer"
    │  └── Returns: final answer with citations
    │
    ▼
Response: answer + community context + token stats
```

**Community level tradeoff:**
- `C0` — 3–5 broadest communities, cheapest (few LLM calls), best for global themes
- `C1` — 15–25 communities, balanced (default)
- `C2` — 50–100 communities, expensive but more granular
- `C3` — 150–300 communities, most expensive, most specific

### VectorRAG Engine

Standard dense retrieval + generation:

```
User Query
    │
    ▼
Embed query using text-embedding-3-small
    │
    ▼
FAISS similarity search → top-K chunks (default K=8)
    │
    ▼
Fill context window: greedily add chunks until token limit
    │
    ▼
LLM: generate answer from retrieved chunks
    │
    ▼
Response: answer + source chunks + token stats
```

---

## Data Models

### Core graph types (`graph_models.py`)

| Model | Purpose |
|---|---|
| `ChunkSchema` | A single text chunk with source metadata |
| `ExtractedEntity` | Named entity extracted from a chunk |
| `ExtractedRelationship` | Relationship between two entities |
| `CommunitySchema` | A community node (id, level, member nodes/edges) |
| `CommunitySummary` | LLM-generated summary of a community |
| `CommunityFinding` | A single finding within a community summary |

### Community levels (`CommunityLevel` enum)

```python
class CommunityLevel(str, Enum):
    C0 = "c0"   # Root: 3–5 broadest communities
    C1 = "c1"   # Level 1: ~15–25 communities (default)
    C2 = "c2"   # Level 2: ~50–100 communities
    C3 = "c3"   # Level 3: ~150–300 communities (most granular)
```

---

## Storage Layout

```
data/
├── raw/                          # Input documents (never modified by pipeline)
│   └── sample_corpus/            # 5 sample AI industry documents
├── processed/                    # Pipeline artifacts (generated, gitignored)
│   ├── chunks.json               # ChunkSchema[]
│   ├── extractions.json          # ExtractionResult[]
│   ├── graph.pkl                 # NetworkX MultiGraph
│   ├── community_map.json        # CommunitySchema[]
│   ├── community_summaries.json  # CommunitySummary[]
│   ├── faiss_index.bin           # FAISS flat index
│   └── embeddings.npy            # numpy float32 array [N, 1536]
└── evaluation/
    ├── questions.json             # Evaluation questions
    └── results.json              # Evaluation results (generated, gitignored)
```

---

## Security

- **API key authentication**: All endpoints require `X-API-Key` header
- **CORS**: Configurable allowed origins via `ALLOWED_ORIGINS` env var
- **Rate limiting**: Configurable per-IP rate limiting via Redis
- **Non-root container**: Backend runs as `appuser` (UID 1001)
- **No secrets in images**: All secrets via environment variables / `.env`

---

## Performance Characteristics

| Operation | Typical latency | Notes |
|---|---|---|
| GraphRAG query (C0) | 3–8s | 3–5 map calls + 1 reduce |
| GraphRAG query (C1) | 8–25s | 15–25 map calls + 1 reduce |
| GraphRAG query (C2) | 30–90s | 50–100 parallel map calls |
| VectorRAG query | 1–3s | 1 embed + 1 generation call |
| Full indexing (100 docs) | 30–90 min | Dominated by extraction LLM calls |

Map-stage calls run in parallel (controlled by semaphore, default 10 concurrent).