"""Load → structure-aware chunk → enrich (context + entities + relations) →
hierarchy (section & document summaries) → index.

Chunk budgets are measured with the embedding model's own tokenizer, so nothing
is silently truncated at embed time. With an API key, one Haiku call per chunk
against a prompt-cached copy of the document returns the contextual-retrieval
sentence (Anthropic 2024) *and* the entity/relation triples for the knowledge
graph. Section and document summaries (RAPTOR, Sarthi et al. 2024) are indexed
as level-1/2 nodes so global questions retrieve a summary, not a random leaf.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from html.parser import HTMLParser
from urllib.request import Request, urlopen

import anthropic
from pypdf import PdfReader

from atlas import config as C
from atlas.store import Store, norm_entity

HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".html", ".csv"}
UPLOADS = C.DATA / "uploads"


@dataclass
class Chunk:
    ordinal: int
    heading: str
    text: str
    start: int  # char offset in the document, for windowed contextualisation
    context: str = ""
    entities: list[str] = field(default_factory=list)
    relations: list[tuple[str, str, str]] = field(default_factory=list)


# ---- load ------------------------------------------------------------------


def load(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        pages = [p.extract_text() or "" for p in PdfReader(str(path)).pages]
        # PDF extractors emit one line per visual line; rejoin into paragraphs.
        return "\n\n".join(re.sub(r"(?<!\n)\n(?!\n)", " ", p).strip() for p in pages)
    if path.suffix.lower() in TEXT_SUFFIXES or not path.suffix:
        return path.read_text(errors="replace")
    raise ValueError(f"unsupported file type: {path.suffix}")


def load_sql(url: str, query: str, name: str) -> Path:
    """Runs `query` against any SQLAlchemy URL and materialises the rows as a
    markdown document (one `##` section per row) under data/uploads, so a table
    goes through exactly the same chunk → enrich → index path as a file."""
    from sqlalchemy import create_engine, text

    with create_engine(url).connect() as conn:
        result = conn.execute(text(query))
        cols = list(result.keys())
        rows = result.fetchall()
    if not rows:
        raise ValueError("query returned no rows")
    out = [f"# {name}\n"]
    for i, row in enumerate(rows, 1):
        title = str(row[0]) if row[0] is not None else f"row {i}"
        out.append(f"## {title}\n" + "\n".join(f"{c}: {v}" for c, v in zip(cols, row) if v not in (None, "")))
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / (re.sub(r"[^\w.-]+", "_", name) + ".md")
    path.write_text("\n\n".join(out))
    return path


class _HtmlToMd(HTMLParser):
    """Stdlib HTML → markdown-ish text: headings become `#` lines (so the chunker
    and hierarchy see structure), block tags become paragraph breaks, chrome is dropped."""

    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "aside", "iframe", "template"}
    BLOCK = {"p", "div", "li", "br", "tr", "section", "article", "blockquote", "pre", "ul", "ol", "table", "main", "hr", "dd", "dt"}

    def __init__(self):
        super().__init__()
        self.out: list[str] = []
        self.title = ""
        self._skip = 0
        self._in = ""  # "title" | "h" | ""

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in = "title"
        elif len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
            self._in = "h"
        elif tag in self.BLOCK:
            self.out.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "title" or self._in == "h" and tag[0] == "h":
            self._in = ""
            self.out.append("\n\n")
        elif tag in self.BLOCK:
            self.out.append("\n\n")

    def handle_data(self, data):
        if self._in == "title":
            self.title += data
        elif not self._skip:
            self.out.append(data.replace("\n", " ") if self._in == "h" else data)


def load_url(url: str, name: str | None = None) -> Path:
    """Fetches a page (or a PDF) and materialises it under data/uploads so it
    takes the same path as a file. ponytail: no JS rendering — add a headless
    browser step if you need SPA content."""
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0 (Atlas RAG)"}), timeout=30) as r:
        raw, ctype = r.read(), r.headers.get_content_type()
    UPLOADS.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w.-]+", "_", name or re.sub(r"^https?://", "", url))[:80]
    if ctype == "application/pdf":
        path = UPLOADS / f"{slug}.pdf"
        path.write_bytes(raw)
        return path
    p = _HtmlToMd()
    p.feed(raw.decode("utf-8", "replace"))
    text = re.sub(r"[ \t]+", " ", "".join(p.out)).replace("¶", "")
    text = re.sub(r"^#+ *$", "", text, flags=re.M)  # headings whose only content was skipped chrome
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    title = " ".join(p.title.split()) or url
    path = UPLOADS / f"{slug}.md"
    path.write_text(f"# {title}\n\nSource: {url}\n\n{text}")
    return path


# ---- chunk -----------------------------------------------------------------


def chunk(doc: str, tokenizer, budget: int = C.CHUNK_TOKENS, overlap: int = C.CHUNK_OVERLAP) -> list[Chunk]:
    ntok = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)  # noqa: E731

    # 1. Split on markdown headings, keeping a breadcrumb of the enclosing ones.
    sections: list[tuple[str, str, int]] = []  # (breadcrumb, body, start_offset)
    crumbs: list[tuple[int, str]] = []
    pos = 0
    for m in [*HEADING.finditer(doc), None]:
        end = m.start() if m else len(doc)
        body = doc[pos:end]
        if body.strip():
            sections.append((" › ".join(c for _, c in crumbs), body, pos))
        if m:
            level = len(m.group(1))
            crumbs = [(l, c) for l, c in crumbs if l < level] + [(level, m.group(2).strip())]
            pos = m.end()

    # 2. Greedy-pack paragraphs (then sentences, then raw tokens) into the budget.
    chunks: list[Chunk] = []
    for crumb, body, base in sections:
        units: list[tuple[str, int]] = []  # (text, offset)
        off = 0
        for raw in re.split(r"\n\s*\n", body):
            para = raw.strip()
            if not para:
                continue
            p_off = body.find(para, off)
            off = p_off + len(para)
            if ntok(para) <= budget:
                units.append((para, base + p_off))
                continue
            s_off = 0
            for sent in SENTENCE.split(para):
                so = para.find(sent, s_off)
                s_off = so + len(sent)
                if ntok(sent) <= budget:
                    units.append((sent, base + p_off + so))
                else:  # pathological run-on: hard split by tokens
                    ids = tokenizer.encode(sent, add_special_tokens=False).ids
                    for i in range(0, len(ids), budget):
                        units.append((tokenizer.decode(ids[i : i + budget]), base + p_off + so))

        buf: list[tuple[str, int]] = []
        size = 0
        for text, offset in units:
            n = ntok(text)
            if buf and size + n > budget:
                chunks.append(Chunk(len(chunks), crumb, "\n\n".join(t for t, _ in buf), buf[0][1]))
                # Overlap: carry trailing units back in until the overlap budget is spent.
                keep, kept = [], 0
                for t, o in reversed(buf):
                    tn = ntok(t)
                    if kept + tn > overlap:
                        break
                    keep.insert(0, (t, o))
                    kept += tn
                buf, size = keep, kept
            buf.append((text, offset))
            size += n
        if buf:
            chunks.append(Chunk(len(chunks), crumb, "\n\n".join(t for t, _ in buf), buf[0][1]))
    return chunks


# ---- enrich: context + knowledge graph, one call per chunk -----------------

ENRICH_PROMPT = (
    "Here is the chunk we want to situate within the whole document:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "1. `context`: 1-2 sentences situating this chunk within the overall document, for the purpose "
    "of improving search retrieval of the chunk. Name the specific entities, sections or topics it belongs to.\n"
    "2. `entities`: the salient named entities and key concepts in the chunk (people, organisations, "
    "products, places, technical terms). Canonical short names, no duplicates, at most 12.\n"
    "3. `relations`: explicit factual relations between those entities stated in the chunk, as "
    "(subject, predicate, object) with a short verb-phrase predicate. At most 10; empty if none."
)
ENRICH_SCHEMA = {
    "type": "object",
    "properties": {
        "context": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"subject": {"type": "string"}, "predicate": {"type": "string"}, "object": {"type": "string"}},
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["context", "entities", "relations"],
    "additionalProperties": False,
}
DOC_WINDOW = 300_000  # chars (~75k tokens) — keeps huge docs inside Haiku's context


async def enrich(doc: str, chunks: list[Chunk], progress=None) -> None:
    """Fills chunk.context / entities / relations in place. Prompt-caches the
    document prefix: the first request per window is awaited alone so it *writes*
    the cache; the rest fan out concurrently and *read* it at 10% of input price."""
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(C.CONTEXT_CONCURRENCY)
    done = 0

    def win_start(ch: Chunk) -> int:
        return 0 if len(doc) <= DOC_WINDOW else max(0, ch.start - DOC_WINDOW // 2)

    async def one(ch: Chunk) -> None:
        nonlocal done
        async with sem:
            s = win_start(ch)
            resp = await client.messages.create(
                model=C.CONTEXT_MODEL,
                max_tokens=800,
                system=[{"type": "text", "text": f"<document>\n{doc[s : s + DOC_WINDOW]}\n</document>", "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": ENRICH_PROMPT.format(chunk=ch.text)}],
                output_config={"format": {"type": "json_schema", "schema": ENRICH_SCHEMA}},
            )
            data = json.loads(next(b.text for b in resp.content if b.type == "text"))
            ch.context = data["context"].strip()
            ch.entities = sorted({norm_entity(e) for e in data["entities"] if norm_entity(e)})
            ch.relations = [
                (norm_entity(r["subject"]), r["predicate"].strip().lower(), norm_entity(r["object"]))
                for r in data["relations"]
                if norm_entity(r["subject"]) and norm_entity(r["object"])
            ]
            done += 1
            if progress:
                await progress("enriching", done, len(chunks), resp.usage)

    groups: dict[int, list[Chunk]] = {}
    for ch in chunks:
        groups.setdefault(win_start(ch), []).append(ch)
    for group in groups.values():
        await one(group[0])  # warm the cache for this window exactly once
        await asyncio.gather(*(one(ch) for ch in group[1:]))


# ---- hierarchy: section + document summaries -------------------------------

SUMMARY_PROMPT = (
    "Summarise the following {what} in 3-6 sentences. Preserve the key entities, numbers, "
    "claims and terminology so the summary is useful for search. Answer with the summary only.\n\n{body}"
)


def group_sections(chunks: list[Chunk]) -> list[tuple[str, list[Chunk]]]:
    """Group leaves under their heading; heading-less runs are windowed."""
    groups: list[tuple[str, list[Chunk]]] = []
    for ch in chunks:
        if groups and groups[-1][0] == ch.heading and (ch.heading or len(groups[-1][1]) < C.SECTION_WINDOW):
            groups[-1][1].append(ch)
        else:
            groups.append((ch.heading, [ch]))
    return groups


async def summarize(groups: list[tuple[str, list[Chunk]]], source: str, progress=None) -> tuple[list[str], str]:
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(C.CONTEXT_CONCURRENCY)
    done = 0

    async def one(what: str, body: str) -> str:
        nonlocal done
        async with sem:
            resp = await client.messages.create(
                model=C.CONTEXT_MODEL,
                max_tokens=600,
                messages=[{"role": "user", "content": SUMMARY_PROMPT.format(what=what, body=body)}],
            )
            done += 1
            if progress:
                await progress("summarizing", done, len(groups) + 1, resp.usage)
            return next((b.text for b in resp.content if b.type == "text"), "").strip()

    sections = await asyncio.gather(
        *(one(f"section '{h or source}'", "\n\n".join(c.text for c in cs)) for h, cs in groups)
    )
    doc_summary = await one(f"document '{source}'", "\n\n".join(sections))
    return list(sections), doc_summary


# ---- pipeline --------------------------------------------------------------


def doc_id_for(path: Path, doc: str) -> str:
    return hashlib.sha1((path.name + doc).encode()).hexdigest()[:16]


async def ingest(path: Path, store: Store, enrich_: bool = True) -> AsyncIterator[dict]:
    """Yields progress events; the last one is {"stage": "done", ...}."""
    doc = await asyncio.to_thread(load, path)
    if not doc.strip():
        raise ValueError(f"{path.name}: no extractable text")
    doc_id = doc_id_for(path, doc)
    yield {"stage": "loaded", "chars": len(doc)}

    chunks = await asyncio.to_thread(chunk, doc, store.tokenizer)
    groups = group_sections(chunks)
    yield {"stage": "chunked", "chunks": len(chunks), "sections": len(groups)}

    usage = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
    enrich_ = enrich_ and C.HAS_API_KEY and len(chunks) >= C.CONTEXT_MIN_CHUNKS and len(doc) >= C.CONTEXT_MIN_DOC_CHARS
    section_summaries: list[str] = []
    doc_summary = ""
    if enrich_:
        queue: asyncio.Queue = asyncio.Queue()

        async def progress(stage, done, total, u):
            usage["input"] += u.input_tokens
            usage["cache_write"] += u.cache_creation_input_tokens or 0
            usage["cache_read"] += u.cache_read_input_tokens or 0
            usage["output"] += u.output_tokens
            await queue.put({"stage": stage, "done": done, "total": total})

        async def work():
            await enrich(doc, chunks, progress)
            return await summarize(groups, path.name, progress)

        task = asyncio.create_task(work())
        while not task.done() or not queue.empty():
            try:
                yield await asyncio.wait_for(queue.get(), 0.25)
            except TimeoutError:
                pass
        section_summaries, doc_summary = task.result()  # re-raises on failure

    # ---- assemble nodes: level 2 (doc) → level 1 (sections) → level 0 (chunks)
    rows: list[dict] = []
    edges: list[dict] = []

    def node(nid, level, parent, heading, context, text, ordinal):
        rows.append(
            {
                "id": nid, "doc_id": doc_id, "source": path.name, "ordinal": ordinal, "level": level, "parent": parent,
                "heading": heading, "context": context, "text": text,
                "embed_text": "\n".join(s for s in (context, heading, text) if s),
            }
        )

    root = f"{doc_id}:doc"
    if doc_summary:
        node(root, 2, "", path.name, "", doc_summary, -1)
    for gi, (heading, members) in enumerate(groups):
        sid = f"{doc_id}:s{gi}"
        if section_summaries:
            node(sid, 1, root if doc_summary else "", heading, "", section_summaries[gi], -1)
        for ch in members:
            cid = f"{doc_id}:{ch.ordinal}"
            node(cid, 0, sid if section_summaries else "", ch.heading, ch.context, ch.text, ch.ordinal)
            for e in ch.entities:
                edges.append({"src": e, "rel": "", "dst": "", "chunk_id": cid, "doc_id": doc_id})
            for s, p, o in ch.relations:
                edges.append({"src": s, "rel": p, "dst": o, "chunk_id": cid, "doc_id": doc_id})

    await asyncio.to_thread(store.delete_doc, doc_id)  # replace, never duplicate
    await asyncio.to_thread(store.add, rows, edges)
    yield {
        "stage": "done", "doc_id": doc_id, "source": path.name, "chunks": len(chunks), "sections": len(groups),
        "nodes": len(rows), "edges": len(edges), "enriched": enrich_, "usage": usage,
    }
