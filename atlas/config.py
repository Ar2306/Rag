"""Single source of truth for every tunable. Env-overridable, no config classes."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("ATLAS_DATA", ROOT / "data"))
DB_PATH = DATA / "lancedb"
TABLE = "chunks"

# --- Models -----------------------------------------------------------------
# bge-small: 384-dim, 512-token window, ONNX/CPU. The quality/latency knee for
# local retrieval. Swap via env; the table is rebuilt on dimension change.
EMBED_MODEL = os.getenv("ATLAS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.getenv("ATLAS_EMBED_DIM", "384"))
# bge-small is asymmetric: queries must carry this prefix, documents must not.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RERANK_MODEL = os.getenv("ATLAS_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

ANSWER_MODEL = os.getenv("ATLAS_ANSWER_MODEL", "claude-opus-5")
ANSWER_EFFORT = os.getenv("ATLAS_ANSWER_EFFORT", "low")  # RAG answers are extractive
CONTEXT_MODEL = os.getenv("ATLAS_CONTEXT_MODEL", "claude-haiku-4-5")  # bulk enricher

# --- Chunking ---------------------------------------------------------------
# Measured in embedding-model tokens, not characters. Leaves headroom in the
# 512-token window for the prepended context sentence + heading breadcrumb.
CHUNK_TOKENS = int(os.getenv("ATLAS_CHUNK_TOKENS", "350"))
CHUNK_OVERLAP = int(os.getenv("ATLAS_CHUNK_OVERLAP", "64"))

# --- Retrieval --------------------------------------------------------------
CANDIDATES = int(os.getenv("ATLAS_CANDIDATES", "50"))  # per arm, before fusion
RRF_K = int(os.getenv("ATLAS_RRF_K", "60"))  # Cormack et al. 2009 default
TOP_K = int(os.getenv("ATLAS_TOP_K", "8"))  # chunks handed to the model
# Knowledge-graph expansion: chunks sharing entities with the query / top hits
# join the rerank pool. Bounded so rerank latency stays flat.
GRAPH_SEED = 10  # top fused hits whose entities seed neighbour expansion
GRAPH_EXPAND = 20  # max neighbour chunks pulled in
RERANK_POOL = 80  # hard cap on candidates the cross-encoder scores

# --- Hierarchy (RAPTOR-style) -------------------------------------------------
# Level 0 = chunks, 1 = section summaries, 2 = document summary. Sections come
# from headings; heading-less docs are windowed into groups of this many chunks.
SECTION_WINDOW = 6

# --- Ingest -----------------------------------------------------------------
CONTEXT_CONCURRENCY = int(os.getenv("ATLAS_CONTEXT_CONCURRENCY", "8"))
# Contextualising a document that fits in one chunk adds cost and no signal.
CONTEXT_MIN_CHUNKS = 2
# Claude's prompt cache has a floor; below it the doc prefix will not cache and
# every chunk pays full input price. Cheaper to skip enrichment entirely.
CONTEXT_MIN_DOC_CHARS = 4_000

HAS_API_KEY = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
