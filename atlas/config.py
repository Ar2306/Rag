"""Single source of truth for every tunable. Env-overridable, no config classes."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("ATLAS_DATA", ROOT / "data"))
DB_PATH = DATA / "lancedb"
TABLE = "chunks"

# --- Local models (ONNX, CPU) ----------------------------------------------
# bge-small: 384-dim, 512-token window. The quality/latency knee for local
# retrieval. Swap via env; the table is rebuilt on dimension change.
EMBED_MODEL = os.getenv("ATLAS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.getenv("ATLAS_EMBED_DIM", "384"))
# bge-small is asymmetric: queries must carry this prefix, documents must not.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RERANK_MODEL = os.getenv("ATLAS_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# --- LLM: any OpenAI-compatible endpoint ------------------------------------
# name: (base_url, answer model, fast/bulk model for enrichment + summaries + rewrites)
# Model names are sensible defaults at the time of writing; override with
# LLM_MODEL / LLM_FAST_MODEL if your provider has renamed them.
PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-pro", "gemini-2.5-flash"),
    "ollama-cloud": ("https://ollama.com/v1", "gpt-oss:120b", "gpt-oss:20b"),
    "ollama": ("http://localhost:11434/v1", "llama3.1:8b", "llama3.1:8b"),  # local, no key
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
    "openrouter": ("https://openrouter.ai/api/v1", "google/gemini-2.5-pro", "google/gemini-2.5-flash"),
    "mistral": ("https://api.mistral.ai/v1", "mistral-large-latest", "mistral-small-latest"),
    "cerebras": ("https://api.cerebras.ai/v1", "llama-3.3-70b", "llama3.1-8b"),
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat", "deepseek-chat"),
    "xai": ("https://api.x.ai/v1", "grok-4", "grok-3-mini"),
    "openai": ("https://api.openai.com/v1", "gpt-5", "gpt-5-mini"),
}
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
if LLM_PROVIDER not in PROVIDERS:
    raise SystemExit(f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)} (got {LLM_PROVIDER!r})")
_base, _answer, _fast = PROVIDERS[LLM_PROVIDER]
LLM_BASE_URL = os.getenv("LLM_BASE_URL", _base)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", _answer)
LLM_FAST_MODEL = os.getenv("LLM_FAST_MODEL", _fast)
HAS_LLM = bool(LLM_API_KEY) or LLM_PROVIDER == "ollama"

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
RERANK_POOL = int(os.getenv("ATLAS_RERANK_POOL", "40"))  # cross-encoder candidates; ~40 ms each on CPU

# --- Hierarchy (RAPTOR-style) -------------------------------------------------
# Level 0 = chunks, 1 = section summaries, 2 = document summary. Sections come
# from headings; heading-less docs are windowed into groups of this many chunks.
SECTION_WINDOW = 6

# --- Ingest -----------------------------------------------------------------
# Free tiers are rate-limited (Gemini Flash ~10-15 RPM on the free plan);
# the SDK retries 429s, but keep this modest.
CONTEXT_CONCURRENCY = int(os.getenv("ATLAS_CONTEXT_CONCURRENCY", "4"))
# Contextualising a document that fits in one chunk adds cost and no signal.
CONTEXT_MIN_CHUNKS = 2
CONTEXT_MIN_DOC_CHARS = 4_000
# Chars of surrounding document shown to the enricher per chunk (~25k tokens):
# fits every provider's context; Gemini caches the repeated prefix implicitly.
DOC_WINDOW = int(os.getenv("ATLAS_DOC_WINDOW", "100000"))
