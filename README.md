# GraphRAG vs VectorRAG

A production-quality implementation of the paper *"From Local to Global: A Graph RAG Approach to Query-Focused Summarization"* (Edge et al., 2024) with a side-by-side comparison against standard VectorRAG.

**Core question:** For global sensemaking queries — questions that require synthesizing information across an entire document corpus — does graph-structured retrieval produce better answers than vector similarity search?

---

## What This Project Does

- **Indexes** a document corpus through a 6-stage pipeline: chunking → entity/relationship extraction → knowledge graph construction → community detection → community summarization → vector embeddings
- **Queries** the indexed corpus with both systems simultaneously and returns side-by-side answers
- **Evaluates** the answers using an LLM-as-judge on 4 criteria: comprehensiveness, diversity, empowerment, directness
- **Visualizes** the knowledge graph, community structure, and evaluation results in a React UI

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- An OpenAI API key

### 1. Clone and configure

```bash
git clone https://github.com/your-org/graphrag-vs-vectorrag
cd graphrag-vs-vectorrag
cp .env.example .env
```

Edit `.env` and set:
```bash
OPENAI_API_KEY=sk-your-key-here
API_KEY=your-strong-random-key   # used to authenticate API requests
```

### 2. Start the stack

```bash
docker-compose up --build -d
```

This starts:
- **Backend** at `http://localhost:8000` (FastAPI + uvicorn, hot reload)
- **Frontend** at `http://localhost:5173` (React + Vite, HMR)
- **Redis** at `localhost:6379`

Check backend health:
```bash
curl http://localhost:8000/api/v1/health
```

### 3. Add documents and index

Copy your `.json` documents to `data/raw/` (or use the included sample corpus):

```bash
# Use sample corpus (5 AI industry documents, ~50K words)
make seed

# Or add your own documents in this format:
# { "text": "Full document text...", "metadata": { ... } }
```

Trigger indexing via the UI (IndexingPage) or CLI:

```bash
# Full pipeline with paper-exact parameters
python scripts/run_indexing.py

# Dev mode: 50 chunks only, no gleaning (fast + cheap)
python scripts/run_indexing.py --max-chunks 50 --gleaning-rounds 0
```

Indexing progress is visible in the UI under the **Indexing** tab.

### 4. Ask global questions

Open `http://localhost:5173` and try questions like:
- *"What are the main themes across the corpus?"*
- *"Who are the most influential organizations?"*
- *"What are the key tensions and disagreements in this field?"*

Or use the API directly:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "What are the major themes in this corpus?",
    "mode": "both",
    "community_level": "c1"
  }'
```

---

## Project Structure

```
graphrag-vs-vectorrag/
├── backend/                    # FastAPI application
│   ├── app/
│   │   ├── api/                # Route handlers
│   │   ├── core/
│   │   │   ├── pipeline/       # Indexing pipeline stages
│   │   │   └── query/          # GraphRAG + VectorRAG engines
│   │   ├── models/             # Pydantic request/response models
│   │   ├── services/           # OpenAI, embeddings, FAISS, tokenizer
│   │   ├── storage/            # Artifact, graph, summary, cache stores
│   │   └── workers/            # Background job workers
│   └── tests/                  # Unit + integration tests
├── frontend/                   # React SPA
│   └── src/
│       ├── api/                # API client layer
│       ├── pages/              # QueryPage, GraphPage, EvaluationPage, IndexingPage
│       ├── components/         # UI components
│       ├── context/            # React context providers
│       └── hooks/              # Custom hooks
├── data/
│   ├── raw/                    # Input documents (add yours here)
│   └── processed/              # Pipeline artifacts (gitignored, generated)
├── scripts/                    # CLI scripts for pipeline stages
├── docs/                       # Architecture and API documentation
└── docker-compose.yml          # Development stack
```

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system design including:
- Indexing pipeline diagram
- GraphRAG map-reduce query pattern
- VectorRAG retrieval pattern
- Community level tradeoffs (C0–C3)
- Storage layout

---

## API Reference

See [docs/api_contract.md](docs/api_contract.md) for all endpoints, or visit `http://localhost:8000/docs` for interactive Swagger UI.

Key endpoints:
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/query` | Query both systems simultaneously |
| `POST` | `/api/v1/index` | Trigger indexing pipeline |
| `GET`  | `/api/v1/index/status` | Poll indexing progress |
| `POST` | `/api/v1/evaluate` | Run LLM-as-judge evaluation |
| `GET`  | `/api/v1/graph` | Knowledge graph statistics |
| `GET`  | `/api/v1/communities/{level}` | Community list at level C0–C3 |

---

## Development

### Local backend (without Docker)

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
cp ../.env.example ../.env   # edit with your keys
uvicorn app.main:app --reload --port 8000
```

### Local frontend (without Docker)

```bash
cd frontend
npm install
npm run dev   # starts at http://localhost:5173
```

### Running tests

```bash
make test               # full suite with coverage
make test-unit          # unit tests only (no network calls)
make test-integration   # integration tests
```

### Code quality

```bash
make lint       # ruff + black check
make format     # auto-format
make typecheck  # mypy
```

### CLI pipeline scripts

```bash
# Run each stage independently
python scripts/run_indexing.py --help
python scripts/run_extraction.py --help
python scripts/run_community_detection.py --help
python scripts/run_summarization.py --help
python scripts/run_evaluation.py --help
```

---

## Production Deployment

```bash
# Build production images + start with nginx reverse proxy
docker-compose -f docker-compose.prod.yml up --build -d

# App will be available at http://localhost:80
```

See [docker-compose.prod.yml](docker-compose.prod.yml) for the full production configuration including nginx, SSL, and Redis with auth.

---

## Evaluation Results

See [docs/evaluation_results.md](docs/evaluation_results.md) for benchmark results.

Run your own evaluation:

```bash
python scripts/run_evaluation.py \
  --questions-file data/evaluation/questions.json \
  --runs 5 \
  --level c1
```

---

## Configuration

All configuration is via environment variables. See [.env.example](.env.example) for the full reference.

Key variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | Your OpenAI API key |
| `API_KEY` | required | Authentication key for API requests |
| `OPENAI_MODEL` | `gpt-4o` | LLM for extraction and queries |
| `CHUNK_SIZE` | `600` | Tokens per chunk (paper: 600) |
| `GLEANING_ROUNDS` | `2` | Self-reflection iterations (paper: 2) |
| `COMMUNITY_LEVEL` | `c1` | Default query level |
| `CONTEXT_WINDOW_SIZE` | `8000` | LLM context window (paper: 8k) |

---

## Paper Reference

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. arXiv:2404.16130.

```bibtex
@article{edge2024graphrag,
  title={From Local to Global: A Graph RAG Approach to Query-Focused Summarization},
  author={Edge, Darren and Trinh, Ha and Cheng, Newman and Bradley, Joshua and
          Chao, Alex and Mody, Apurva and Truitt, Steven and Larson, Jonathan},
  journal={arXiv preprint arXiv:2404.16130},
  year={2024}
}
```

---

## License

[MIT](LICENSE)