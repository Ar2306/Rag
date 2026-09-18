"""Index + retrieval. One LanceDB database holds vectors, the BM25 index, the
hierarchy (level/parent columns) and the entity graph, so nothing can drift.

Per query:  embed → dense top-C ∥ BM25 top-C → RRF fuse → graph expansion →
cross-encoder rerank → top-K.  Every stage is timed; timings ride with the hits.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

import lancedb
import numpy as np
import pyarrow as pa
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from atlas import config as C

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("ordinal", pa.int32()),
        pa.field("level", pa.int8()),  # 0 chunk · 1 section summary · 2 doc summary
        pa.field("parent", pa.string()),  # id of the node one level up ("" at root)
        pa.field("heading", pa.string()),
        pa.field("context", pa.string()),
        pa.field("text", pa.string()),  # what the LLM reads
        pa.field("embed_text", pa.string()),  # context + heading + text — what we search
        pa.field("vector", pa.list_(pa.float32(), C.EMBED_DIM)),
    ]
)
GRAPH_SCHEMA = pa.schema(
    [
        pa.field("src", pa.string()),  # normalised entity name
        pa.field("rel", pa.string()),  # "" for a bare mention
        pa.field("dst", pa.string()),  # "" for a bare mention
        pa.field("chunk_id", pa.string()),
        pa.field("doc_id", pa.string()),
    ]
)
COLS = ["id", "doc_id", "source", "ordinal", "level", "parent", "heading", "context", "text"]
MODES = ("dense", "bm25", "hybrid", "hybrid+rerank", "full")  # full = hybrid + graph + rerank


@dataclass
class Hit:
    id: str
    doc_id: str
    source: str
    ordinal: int
    level: int
    parent: str
    heading: str
    context: str
    text: str
    score: float = 0.0  # final-stage score (rerank logit, or RRF score)
    arms: list[str] = field(default_factory=list)  # which retrieval arms found it

    def as_dict(self) -> dict:
        return self.__dict__


def rrf(*rankings: list[str], k: int = C.RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009)."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, id_ in enumerate(ranking):
            scores[id_] += 1.0 / (k + rank + 1)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def norm_entity(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def _sql_list(values) -> str:
    return ", ".join(_sql_str(v) for v in values)


def _sql_str(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


class Store:
    def __init__(self) -> None:
        self.embedder = TextEmbedding(C.EMBED_MODEL)
        self.reranker = TextCrossEncoder(C.RERANK_MODEL)
        self.tokenizer = self.embedder.model.tokenizer
        self.db = lancedb.connect(str(C.DB_PATH))
        self.table = self._open(C.TABLE, SCHEMA)
        self.graph = self._open("graph", GRAPH_SCHEMA)
        self._entities: dict[str, re.Pattern] = {}
        self._refresh()
        # Warm ONNX sessions so the first user query does not pay JIT/alloc cost.
        list(self.embedder.query_embed("warmup"))
        list(self.reranker.rerank("warmup", ["warmup"]))

    def _open(self, name, schema):
        return self.db.create_table(name, schema=schema, exist_ok=True)  # opens if present

    # ---- write --------------------------------------------------------------

    def add(self, rows: list[dict], edges: list[dict]) -> None:
        if rows:
            # In-process ONNX threads. `parallel=0` would fork one model-loading
            # worker per core for any doc with >= 64 chunks and OOM small machines.
            vectors = self.embedder.embed([r["embed_text"] for r in rows], batch_size=64)
            for r, v in zip(rows, vectors):
                r["vector"] = np.asarray(v, dtype=np.float32)
            self.table.add(pa.Table.from_pylist(rows, schema=SCHEMA))
        if edges:
            self.graph.add(pa.Table.from_pylist(edges, schema=GRAPH_SCHEMA))
        self._refresh()

    def delete_doc(self, doc_id: str) -> None:
        self.table.delete(f"doc_id = {_sql_str(doc_id)}")
        self.graph.delete(f"doc_id = {_sql_str(doc_id)}")
        self._refresh()

    def _refresh(self) -> None:
        # ponytail: full FTS rebuild on every write. Milliseconds up to ~100k
        # rows; switch to optimize()-merged incremental indexing beyond that.
        n = self.table.count_rows()
        self._fts_ready = n > 0
        if self._fts_ready:
            self.table.create_fts_index("embed_text", replace=True)
        # Flat (exact) scan beats an ANN index below tens of thousands of rows.
        if n >= 20_000:
            self.table.create_index(metric="cosine", index_type="IVF_PQ", replace=True)
        # Entity vocabulary for query matching. ponytail: linear regex scan per
        # query — fine to ~50k entities, Aho-Corasick beyond that.
        names = set()
        if self.graph.count_rows():
            g = self.graph.search().select(["src", "dst"]).limit(10_000_000).to_arrow()
            names = {n for col in ("src", "dst") for n in g[col].to_pylist() if len(n) >= 3}
        self._entities = {n: re.compile(r"\b" + re.escape(n) + r"\b", re.I) for n in names}

    # ---- read ---------------------------------------------------------------

    def count(self) -> int:
        return self.table.count_rows()

    def docs(self) -> list[dict]:
        if not self.table.count_rows():
            return []
        t = self.table.search().select(["doc_id", "source", "level"]).limit(10_000_000).to_arrow()
        agg: dict[tuple[str, str], dict] = {}
        for d, s, lv in zip(t["doc_id"].to_pylist(), t["source"].to_pylist(), t["level"].to_pylist()):
            a = agg.setdefault((d, s), {"doc_id": d, "source": s, "chunks": 0, "sections": 0, "summary": False})
            if lv == 0:
                a["chunks"] += 1
            elif lv == 1:
                a["sections"] += 1
            else:
                a["summary"] = True
        g = self.graph.search().select(["doc_id"]).limit(10_000_000).to_arrow()["doc_id"].to_pylist() if self.graph.count_rows() else []
        edge_counts = defaultdict(int)
        for d in g:
            edge_counts[d] += 1
        for (d, _), a in agg.items():
            a["edges"] = edge_counts[d]
        return sorted(agg.values(), key=lambda a: a["source"])

    def tree(self, doc_id: str) -> list[dict]:
        rows = self.table.search().where(f"doc_id = {_sql_str(doc_id)}").select(COLS).limit(1_000_000).to_list()
        return sorted(rows, key=lambda r: (-r["level"], r["ordinal"]))

    def graph_edges(self, doc_id: str | None = None, limit: int = 400) -> list[dict]:
        if not self.graph.count_rows():
            return []
        q = self.graph.search().where("rel != ''")
        if doc_id:
            q = q.where(f"doc_id = {_sql_str(doc_id)}")
        return q.select(["src", "rel", "dst", "chunk_id"]).limit(limit).to_list()

    def _chunks_with_entities(self, names: set[str], exclude: set[str], limit: int) -> list[str]:
        if not names:
            return []
        lst = _sql_list(names)
        rows = self.graph.search().where(f"src IN ({lst}) OR dst IN ({lst})").select(["chunk_id"]).limit(5000).to_list()
        # Rank neighbours by how many query/seed entities they share.
        counts = defaultdict(int)
        for r in rows:
            if r["chunk_id"] not in exclude:
                counts[r["chunk_id"]] += 1
        return [c for c, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]]

    def _entities_of(self, chunk_ids: list[str]) -> set[str]:
        if not chunk_ids:
            return set()
        rows = self.graph.search().where(f"chunk_id IN ({_sql_list(chunk_ids)})").select(["src", "dst"]).limit(5000).to_list()
        return {n for r in rows for n in (r["src"], r["dst"]) if n}

    def _fetch(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        rows = self.table.search().where(f"id IN ({_sql_list(ids)})").select(COLS).limit(len(ids)).to_list()
        by = {r["id"]: r for r in rows}
        return [by[i] for i in ids if i in by]

    def retrieve(
        self, query: str, k: int = C.TOP_K, mode: str = "full", candidates: int = C.CANDIDATES,
        doc_ids: list[str] | None = None,
    ) -> tuple[list[Hit], dict[str, float], dict]:
        """Returns (hits, stage timings in ms, meta). `doc_ids` scopes every arm
        to those documents (pre-filtered, so top-C is still C *within* scope)."""
        assert mode in MODES, mode
        meta: dict = {"mode": mode, "entities": []}
        if not self._fts_ready:
            return [], {}, meta
        t: dict[str, float] = {}
        rows: dict[str, dict] = {}
        scope = f"doc_id IN ({_sql_list(doc_ids)})" if doc_ids else None

        def _scoped(q):
            return q.where(scope, prefilter=True) if scope else q

        def _run(name: str, fn):
            t0 = time.perf_counter()
            out = fn()
            t[name] = round((time.perf_counter() - t0) * 1000, 2)
            return out

        def _absorb(found: list[dict], arm: str) -> list[str]:
            for r in found:
                rows.setdefault(r["id"], r).setdefault("arms", []).append(arm)
            return [r["id"] for r in found]

        rankings: list[list[str]] = []
        if mode != "bm25":
            qvec = _run("embed", lambda: next(iter(self.embedder.query_embed(query))))
            dense = _run(
                "dense",
                lambda: _scoped(self.table.search(qvec, vector_column_name="vector").metric("cosine")).select(COLS).limit(candidates).to_list(),
            )
            rankings.append(_absorb(dense, "dense"))
        if mode != "dense":
            bm25 = _run(
                "bm25",
                lambda: _scoped(self.table.search(query, query_type="fts", fts_columns="embed_text")).select(COLS).limit(candidates).to_list(),
            )
            rankings.append(_absorb(bm25, "bm25"))

        fused = _run("fuse", lambda: rrf(*rankings))
        order = list(fused)

        if mode == "full":

            def _expand():
                # (a) entities named in the query itself  (b) entities of the top seeds
                in_query = {n for n, pat in self._entities.items() if pat.search(query)}
                seeds = self._entities_of(order[: C.GRAPH_SEED])
                direct = self._chunks_with_entities(in_query, set(), C.GRAPH_EXPAND)
                for i in direct:  # already-retrieved chunks still earn the badge
                    if i in rows:
                        rows[i]["arms"].append("graph")
                new = [i for i in direct if i not in rows]
                have = set(order) | set(new)
                neigh = self._chunks_with_entities(seeds - in_query, have, max(0, C.GRAPH_EXPAND - len(new)))
                fetched = [r for r in self._fetch(new + neigh) if not doc_ids or r["doc_id"] in doc_ids]
                return _absorb(fetched, "graph"), sorted(in_query)

            extra, meta["entities"] = _run("graph", _expand)
            order = (order + extra)[: C.RERANK_POOL]

        if mode in ("hybrid+rerank", "full") and order:
            scores = _run("rerank", lambda: list(self.reranker.rerank(query, [rows[i]["text"] for i in order])))
            ranked = sorted(zip(order, scores), key=lambda x: x[1], reverse=True)
        else:
            ranked = [(i, fused.get(i, 0.0)) for i in order]

        hits = [Hit(**{c: rows[i][c] for c in COLS}, score=float(s), arms=rows[i]["arms"]) for i, s in ranked[:k]]
        t["total"] = round(sum(t.values()), 2)
        meta["pool"] = len(order)
        return hits, t, meta
