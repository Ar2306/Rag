"""FastAPI surface. Everything long-running streams as SSE so the UI can show
stage-by-stage progress and token-by-token answers."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from atlas import config as C
from atlas.answer import standalone_question, stream_answer
from atlas.ingest import UPLOADS, ingest, load_sql, load_url
from atlas.store import MODES, Store

WEB = C.ROOT / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = await asyncio.to_thread(Store)
    app.state.write_lock = asyncio.Lock()  # LanceDB writes are serialised
    yield


app = FastAPI(title="Atlas RAG", lifespan=lifespan)


def sse(gen: AsyncIterator[dict]) -> StreamingResponse:
    async def body():
        try:
            async for ev in gen:
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:  # surface to the client instead of a dropped stream
            yield f"data: {json.dumps({'stage': 'error', 't': 'error', 'error': f'{type(e).__name__}: {e}'})}\n\n"

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---- static + status ---------------------------------------------------------


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html")


@app.get("/health")
async def health(req: Request):
    s: Store = req.app.state.store
    nodes, docs = await asyncio.to_thread(lambda: (s.count(), len(s.docs())))
    return {
        "nodes": nodes, "docs": docs, "api_key": C.HAS_LLM, "provider": C.LLM_PROVIDER,
        "models": {"embed": C.EMBED_MODEL, "rerank": C.RERANK_MODEL, "answer": C.LLM_MODEL, "enrich": C.LLM_FAST_MODEL},
        "modes": MODES,
    }


# ---- documents ---------------------------------------------------------------


@app.get("/documents")
async def list_docs(req: Request):
    return await asyncio.to_thread(req.app.state.store.docs)


@app.get("/documents/{doc_id}/tree")
async def doc_tree(doc_id: str, req: Request):
    return await asyncio.to_thread(req.app.state.store.tree, doc_id)


@app.delete("/documents/{doc_id}")
async def delete_doc(doc_id: str, req: Request):
    async with req.app.state.write_lock:
        await asyncio.to_thread(req.app.state.store.delete_doc, doc_id)
    return {"ok": True}


@app.get("/graph")
async def graph(req: Request, doc_id: str | None = None, limit: int = 400):
    return await asyncio.to_thread(req.app.state.store.graph_edges, doc_id, limit)


# ---- ingest ------------------------------------------------------------------


async def _ingest_paths(req: Request, paths: list[Path], enrich: bool) -> AsyncIterator[dict]:
    store: Store = req.app.state.store
    for p in paths:
        try:
            async with req.app.state.write_lock:
                async for ev in ingest(p, store, enrich):
                    yield {"file": p.name, **ev}
        except Exception as e:
            yield {"file": p.name, "stage": "error", "error": f"{type(e).__name__}: {e}"}


@app.post("/ingest")
async def ingest_files(req: Request, files: list[UploadFile] = File(...), enrich: bool = Form(True)):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        p = UPLOADS / Path(f.filename or "upload").name
        with p.open("wb") as out:  # stream to disk; never hold a whole PDF in memory
            await asyncio.to_thread(shutil.copyfileobj, f.file, out)
        paths.append(p)
    return sse(_ingest_paths(req, paths, enrich))


class SqlIngest(BaseModel):
    url: str = Field(examples=["sqlite:///data/example.db", "postgresql+psycopg://user:pw@host/db"])
    query: str = Field(examples=["SELECT title, body FROM articles"])
    name: str = Field(examples=["articles"])
    enrich: bool = True


@app.post("/ingest/sql")
async def ingest_sql(req: Request, body: SqlIngest):
    try:
        path = await asyncio.to_thread(load_sql, body.url, body.query, body.name)
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return sse(_ingest_paths(req, [path], body.enrich))


class UrlIngest(BaseModel):
    url: str = Field(examples=["https://docs.example.com/guide"])
    name: str | None = None
    enrich: bool = True


@app.post("/ingest/url")
async def ingest_url(req: Request, body: UrlIngest):
    try:
        path = await asyncio.to_thread(load_url, body.url, body.name)
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    return sse(_ingest_paths(req, [path], body.enrich))


# ---- query -------------------------------------------------------------------


class Query(BaseModel):
    question: str
    mode: str = "full"
    k: int = Field(C.TOP_K, ge=1, le=C.RERANK_POOL)
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": str}]
    doc_ids: list[str] | None = None  # scope retrieval to these documents


@app.post("/query")
async def query(req: Request, q: Query):
    if q.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    store: Store = req.app.state.store

    async def gen():
        t0 = time.perf_counter()
        history = q.history[-6:]
        question = await standalone_question(q.question, history) if history and C.HAS_LLM else q.question
        if question != q.question:
            yield {"t": "rewrite", "question": question, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        hits, timings, meta = await asyncio.to_thread(store.retrieve, question, q.k, q.mode, C.CANDIDATES, q.doc_ids)
        yield {"t": "retrieval", "hits": [h.as_dict() for h in hits], "timings": timings, **meta}
        if not hits:
            yield {"t": "error", "error": "No documents indexed yet — upload something first."}
            return
        if not C.HAS_LLM:
            yield {"t": "error", "error": "No LLM configured (set LLM_PROVIDER and LLM_API_KEY in .env): retrieval works, answer generation is disabled."}
            return
        first = None
        async for ev in stream_answer(q.question, hits, history):
            if ev["t"] == "text" and first is None:
                first = (time.perf_counter() - t0) * 1000
                yield {"t": "ttft", "ms": round(first, 1)}
            if ev["t"] == "done":
                ev["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            yield ev

    return sse(gen())
