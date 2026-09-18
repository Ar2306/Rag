# Atlas

Local-first RAG that is built the way the retrieval literature says to build it, not the way tutorials do.

```
upload / SQL ──► structure-aware chunks ──► Haiku enrich (context + entities + relations, prompt-cached)
                                        ──► section & document summaries (hierarchy)
                                        ──► LanceDB: dense vectors + BM25 + entity graph

question ──► embed ──► dense top-50 ─┐
         ──► BM25  top-50 ───────────┼──► RRF (k=60) ──► graph expansion ──► cross-encoder rerank ──► top-8
                                     │
         ──► entity match ───────────┘

top-8 ──► Claude (custom-content documents, native citations) ──► streamed answer + exact quotes
```

## Why these choices

| Decision | Evidence |
|---|---|
| **Hybrid BM25 + dense, fused with RRF** | Hybrid beats either arm on every benchmark in [arXiv 2604.01733](https://arxiv.org/html/2604.01733v1): Recall@5 0.695 vs 0.644 (BM25) / 0.587 (dense). BM25 alone beats dense on domain text — never drop lexical. |
| **Cross-encoder rerank of the fused pool** | Largest single gain in the same study: +12pp Recall@5, MRR@3 0.433 → 0.605. |
| **Contextual chunk enrichment** | Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval): −49% retrieval failures, −67% with rerank. Prefix is prompt-cached so it costs ≈ $1 / M document tokens. |
| **No HyDE, no CRAG** | Both *underperform* plain hybrid (HyDE Recall@5 0.544) — query expansion hurts precise/numeric questions. |
| **Hierarchical nodes (RAPTOR-style)** | Leaf chunks can't answer "what is this document about". Section + document summaries are indexed alongside leaves; the reranker picks the right altitude. |
| **Entity graph expansion** | Chunks that share entities with the query or the top hits join the rerank pool — catches multi-hop and "who/what is X" questions that neither arm scores well. |
| **Native API citations** | Each chunk is a custom-content document; Claude returns `document_index` + `cited_text`. No regex over `[3]`, no invented references. |
| **Standalone-question rewrite for follow-ups** | "what caused it?" retrieves nothing on its own. Haiku resolves references using the conversation — coreference resolution, *not* query expansion, so it adds nothing the chat didn't contain. |
| **Document scoping** | Tick documents in the sidebar to pre-filter every retrieval arm (top-C is still C within scope). |
| **LanceDB embedded** | Vectors, BM25 index and graph in one process, ~4 MB idle. No server to run or keep in sync. |
| **fastembed (ONNX)** | bge-small (384-d) + MiniLM cross-encoder on CPU: full retrieval in ~30–60 ms on an M1. |

Retrieval works with **no API key** (upload → hybrid search → sources). Enrichment, summaries, graph and answers need `ANTHROPIC_API_KEY`.

## Run

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY
uv sync
uv run uvicorn atlas.server:app --port 8000
open http://localhost:8000
```

First start downloads the two ONNX models (~200 MB total) once.

Docker: `docker build -t atlas . && docker run -p 8000:8000 -v atlas-data:/data -e ANTHROPIC_API_KEY atlas`

## Ingest

Three sources, one pipeline: **files** (pdf · md · txt · code, drag-and-drop), **URLs** (HTML pages become markdown with headings preserved; PDF links are fetched as PDFs), and **databases** below.

## Ingest from any database

The **Connect a database** panel (or `POST /ingest/sql`) takes any SQLAlchemy URL and a query. Each row becomes a section, so it flows through the same chunk → enrich → index path as a file.

```json
{ "url": "postgresql+psycopg://user:pw@host/db", "query": "SELECT title, body FROM articles", "name": "articles" }
```

SQLite works out of the box; install the driver for anything else (`uv add psycopg`, `pymysql`, …).

## Prove it on your corpus

```bash
uv run python -m atlas.evaluate build 40   # Haiku writes 40 gold questions from random chunks
uv run python -m atlas.evaluate run        # Recall@k / MRR / nDCG / p50 latency per mode
```

Every mode in the UI dropdown (`dense`, `bm25`, `hybrid`, `hybrid+rerank`, `full`) is scored, so the ablation is one command away.

## Self-check

```bash
uv run python tests/test_atlas.py
```

## Layout

```
atlas/config.py     every tunable, env-overridable
atlas/ingest.py     load (pdf/text/SQL) → chunk → enrich → summarise → index
atlas/store.py      LanceDB tables, RRF, graph expansion, rerank, timings
atlas/answer.py     Claude streaming with native citations
atlas/server.py     FastAPI + SSE
atlas/evaluate.py   IR eval harness
web/index.html      UI: chat, sources, latency waterfall, knowledge graph, hierarchy
```
