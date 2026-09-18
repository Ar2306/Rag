# Atlas

Local-first RAG that is built the way the retrieval literature says to build it, not the way tutorials do.

```
upload / SQL ──► structure-aware chunks ──► LLM enrich (context + entities + relations, JSON mode)
                                        ──► section & document summaries (hierarchy)
                                        ──► LanceDB: dense vectors + BM25 + entity graph

question ──► embed ──► dense top-50 ─┐
         ──► BM25  top-50 ───────────┼──► RRF (k=60) ──► graph expansion ──► cross-encoder rerank ──► top-8
                                     │
         ──► entity match ───────────┘

top-8 ──► LLM (numbered sources) ──► streamed answer with [n] citations → clickable source chips
```

## Why these choices

| Decision | Evidence |
|---|---|
| **Hybrid BM25 + dense, fused with RRF** | Hybrid beats either arm on every benchmark in [arXiv 2604.01733](https://arxiv.org/html/2604.01733v1): Recall@5 0.695 vs 0.644 (BM25) / 0.587 (dense). BM25 alone beats dense on domain text — never drop lexical. |
| **Cross-encoder rerank of the fused pool** | Largest single gain in the same study: +12pp Recall@5, MRR@3 0.433 → 0.605. |
| **Contextual chunk enrichment** | [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval): −49% retrieval failures, −67% with rerank. Each chunk is enriched with a window of its document; Gemini caches the repeated prefix implicitly. |
| **No HyDE, no CRAG** | Both *underperform* plain hybrid (HyDE Recall@5 0.544) — query expansion hurts precise/numeric questions. |
| **Hierarchical nodes (RAPTOR-style)** | Leaf chunks can't answer "what is this document about". Section + document summaries are indexed alongside leaves; the reranker picks the right altitude. |
| **Entity graph expansion** | Chunks that share entities with the query or the top hits join the rerank pool — catches multi-hop and "who/what is X" questions that neither arm scores well. |
| **Numbered citations** | Each chunk is a numbered source; the model cites `[n]` and the stream is parsed into clickable chips as markers complete. Out-of-range numbers are dropped, so a citation always points at a real source. Works on every provider. |
| **Standalone-question rewrite for follow-ups** | "what caused it?" retrieves nothing on its own. The fast model resolves references using the conversation — coreference resolution, *not* query expansion, so it adds nothing the chat didn't contain. |
| **Document scoping** | Tick documents in the sidebar to pre-filter every retrieval arm (top-C is still C within scope). |
| **LanceDB embedded** | Vectors, BM25 index and graph in one process, ~4 MB idle. No server to run or keep in sync. |
| **fastembed (ONNX)** | bge-small (384-d) + MiniLM cross-encoder on CPU: full retrieval in ~30–60 ms on an M1. |

Retrieval works with **no LLM at all** (upload → hybrid search → sources). Enrichment, summaries, graph and answers need one.

## Choose an LLM provider

Every provider below exposes an OpenAI-compatible endpoint, so the app has one LLM code path. Set `LLM_PROVIDER` and `LLM_API_KEY` in `.env`; `LLM_MODEL` / `LLM_FAST_MODEL` override the defaults (answer model / bulk model for enrichment, summaries and rewrites).

| `LLM_PROVIDER` | Free tier | Notes |
|---|---|---|
| `gemini` (default) | yes, rate-limited | 1M context, JSON mode, implicit prefix caching — best for enrichment |
| `ollama-cloud` | yes, usage limits | big open models (gpt-oss, Qwen, DeepSeek) on Ollama's servers, same API as local |
| `ollama` | free, local | no key; needs `ollama serve` and a pulled model. Small models only on 8 GB |
| `groq` | yes | fastest inference; open models |
| `openrouter` | free-tagged models | one key, every vendor |
| `mistral` | experimental tier | JSON mode, EU-hosted |
| `cerebras` | yes | very fast open models |
| `deepseek` | no, cheap | strong reasoning, JSON mode |
| `xai` | no | Grok |
| `openai` | no | GPT |

Model names in `atlas/config.py` are defaults at the time of writing — check your provider's model list if a request 404s.

## Run

```bash
cp .env.example .env            # set LLM_PROVIDER + LLM_API_KEY
uv sync
uv run uvicorn atlas.server:app --port 8000
open http://localhost:8000
```

First start downloads the two ONNX models (~200 MB total) once.

Docker: `docker build -t atlas . && docker run -p 8000:8000 -v atlas-data:/data -e LLM_PROVIDER -e LLM_API_KEY atlas`

## Ingest

Three sources, one pipeline: **files** (drag-and-drop), **URLs** (HTML pages become markdown with headings preserved; PDF links are fetched as PDFs), and **databases** below.

| File type | Loader |
|---|---|
| `.pdf` | PyMuPDF via `pymupdf4llm`: multi-column reading order, headings inferred from font sizes. Pages without a text layer are OCR'd with RapidOCR (ONNX, CPU). |
| `.docx` | stdlib zip + XML; `Heading N` styles become markdown headings |
| `.ipynb` | markdown cells verbatim, code cells fenced, outputs dropped |
| `.md` `.txt` `.rst` `.html` | as-is (markdown headings drive the hierarchy) |
| `.py` `.js` `.ts` `.json` `.yaml` `.csv` | fenced as code, so `# comment` lines are not mistaken for headings |

## Ingest from any database

The **Connect a database** panel (or `POST /ingest/sql`) takes any SQLAlchemy URL and a query. Each row becomes a section, so it flows through the same chunk → enrich → index path as a file.

```json
{ "url": "postgresql+psycopg://user:pw@host/db", "query": "SELECT title, body FROM articles", "name": "articles" }
```

SQLite works out of the box; install the driver for anything else (`uv add psycopg`, `pymysql`, …).

## Prove it on your corpus

```bash
uv run python -m atlas.evaluate build 40   # the fast model writes 40 gold questions from random chunks
uv run python -m atlas.evaluate run        # Recall@k / MRR / nDCG / p50 latency per mode
```

Every mode in the UI dropdown (`dense`, `bm25`, `hybrid`, `hybrid+rerank`, `full`) is scored, so the ablation is one command away.

## Self-check

```bash
uv run python tests/test_atlas.py
```

## Layout

```
atlas/config.py     every tunable + provider presets, env-overridable
atlas/llm.py        one OpenAI-compatible client for every provider
atlas/ingest.py     load (pdf/docx/ipynb/text/SQL) → chunk → enrich → summarise → index
atlas/store.py      LanceDB tables, RRF, graph expansion, rerank, timings
atlas/answer.py     streamed answer with [n] citations parsed into source chips
atlas/server.py     FastAPI + SSE
atlas/evaluate.py   IR eval harness
web/index.html      UI: chat, sources, latency waterfall, knowledge graph, hierarchy
```
