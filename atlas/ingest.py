"""Load → structure-aware chunk → enrich (context + entities + relations) →
hierarchy (section & document summaries) → index.

Chunk budgets are measured with the embedding model's own tokenizer, so nothing
is silently truncated at embed time. With an LLM configured, one fast-model call
per chunk (JSON mode, with a window of the surrounding document) returns the
contextual-retrieval sentence (Anthropic 2024) *and* the entity/relation triples
for the knowledge graph. Section and document summaries (RAPTOR, Sarthi et al.
2024) are indexed as level-1/2 nodes so global questions retrieve a summary,
not a random leaf.
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

from atlas import config as C
from atlas.llm import complete, parse_json
from atlas.store import Store, norm_entity

HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)
FENCE = re.compile(r"^```.*?^```[ \t]*$", re.M | re.S)  # headings inside code fences are not headings
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".html"}
CODE_SUFFIXES = {".py", ".js", ".ts", ".json", ".yaml", ".yml", ".csv"}  # fenced, so `# comment` lines stay code
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
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


_ocr = None


def _ocr_page(page) -> str:
    """OCR one rasterised page with RapidOCR (ONNX, ~80 MB, no torch). Lazy so
    text-layer PDFs never pay for it."""
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr = RapidOCR()
    import numpy as np

    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    result, _ = _ocr(img)
    if not result:
        return ""
    # Reading order: top-to-bottom, then left-to-right, by box top-left corner.
    lines = sorted(result, key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))
    return "\n".join(r[1] for r in lines)


OCR_MIN_CHARS = 40  # a text-layer page with fewer chars than this is treated as an image


def load_pdf(path: Path) -> str:
    """PyMuPDF via pymupdf4llm: correct multi-column reading order and headings
    inferred from font sizes (emitted as markdown `#`), so PDFs get the same
    hierarchy as markdown. Pages without a text layer fall back to OCR."""
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(str(path))
    # use_ocr=False: pymupdf4llm's own RapidOCR hook targets an older RapidOCR API
    # and crashes on 1.4+; we OCR image-only pages ourselves below.
    pages = pymupdf4llm.to_markdown(doc, page_chunks=True, table_strategy=None, show_progress=False, use_ocr=False)
    out = []
    for i, p in enumerate(pages):
        text = p["text"].strip()
        if len(text) < OCR_MIN_CHARS:
            try:
                text = _ocr_page(doc[i]).strip()
            except Exception:  # OCR is best-effort: missing package, odd image, etc.
                pass
        if text:
            out.append(text)
    text = "\n\n".join(out)
    text = re.sub(r"</?mark>", "", text)  # pymupdf4llm highlight markup
    return re.sub(r"^(#{1,6}) \*\*(.+?)\*\*\s*$", r"\1 \2", text, flags=re.M)  # bold headings → plain


def load_ipynb(path: Path) -> str:
    """Markdown cells as-is, code cells fenced; outputs are dropped."""
    out = []
    for cell in json.loads(path.read_text(errors="replace")).get("cells", []):
        src = cell.get("source", "")
        src = (src if isinstance(src, str) else "".join(src)).strip()
        if src:
            out.append(src if cell.get("cell_type") == "markdown" else f"```\n{src}\n```")
    return "\n\n".join(out)


def load_docx(path: Path) -> str:
    """Stdlib .docx reader: one line per paragraph, Heading N styles become `#` headings."""
    import zipfile
    from xml.etree import ElementTree as ET

    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    out = []
    for p in root.iter(f"{W}p"):
        text = "".join(t.text or "" for t in p.iter(f"{W}t")).strip()
        if not text:
            continue
        style = p.find(f"{W}pPr/{W}pStyle")
        m = re.fullmatch(r"Heading(\d)", style.get(f"{W}val", "") if style is not None else "")
        out.append(f"{'#' * int(m.group(1))} {text}" if m else text)
    return "\n\n".join(out)


def load(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".ipynb":
        return load_ipynb(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix in CODE_SUFFIXES:
        return f"```\n{path.read_text(errors='replace')}\n```"
    if suffix in TEXT_SUFFIXES or not suffix:
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
    fences = [(m.start(), m.end()) for m in FENCE.finditer(doc)]
    headings = [m for m in HEADING.finditer(doc) if not any(a <= m.start() < b for a, b in fences)]
    for m in [*headings, None]:
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
                else:  # pathological run-on: hard split on token boundaries, slicing the *original* text
                    enc = tokenizer.encode(sent, add_special_tokens=False)  # decode() would lowercase/normalise
                    for i in range(0, len(enc.ids), budget):
                        a, b = enc.offsets[i][0], enc.offsets[min(i + budget, len(enc.ids)) - 1][1]
                        units.append((sent[a:b], base + p_off + so + a))

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
    "Return a JSON object with exactly these keys:\n"
    '"context": 1-2 sentences situating this chunk within the overall document, for the purpose of '
    "improving search retrieval of the chunk. Name the specific entities, sections or topics it belongs to.\n"
    '"entities": array of the salient named entities and key concepts in the chunk (people, organisations, '
    "products, places, technical terms). Canonical short names, no duplicates, at most 12.\n"
    '"relations": array of {{"subject", "predicate", "object"}} objects for explicit factual relations '
    "between those entities stated in the chunk, with a short verb-phrase predicate. At most 10; empty if none."
)


async def enrich(doc: str, chunks: list[Chunk], progress=None) -> None:
    """Fills chunk.context / entities / relations in place. Each call sees a window
    of the document around the chunk (the whole document when it fits)."""
    sem = asyncio.Semaphore(C.CONTEXT_CONCURRENCY)
    done = 0

    async def one(ch: Chunk) -> None:
        nonlocal done
        async with sem:
            s = 0 if len(doc) <= C.DOC_WINDOW else max(0, ch.start - C.DOC_WINDOW // 2)
            raw, usage = await complete(
                C.LLM_FAST_MODEL,
                [
                    {"role": "system", "content": f"<document>\n{doc[s : s + C.DOC_WINDOW]}\n</document>"},
                    {"role": "user", "content": ENRICH_PROMPT.format(chunk=ch.text)},
                ],
                json_mode=True,
            )
            data = parse_json(raw)
            ch.context = str(data.get("context") or "").strip()
            ch.entities = sorted({norm_entity(str(e)) for e in data.get("entities") or [] if norm_entity(str(e))})
            ch.relations = []
            for r in data.get("relations") or []:
                if not isinstance(r, dict):
                    continue
                subj, obj = norm_entity(str(r.get("subject") or "")), norm_entity(str(r.get("object") or ""))
                if subj and obj:
                    ch.relations.append((subj, str(r.get("predicate") or "").strip().lower(), obj))
            done += 1
            if progress:
                await progress("enriching", done, len(chunks), usage)

    await asyncio.gather(*(one(ch) for ch in chunks))


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
    sem = asyncio.Semaphore(C.CONTEXT_CONCURRENCY)
    done = 0

    async def one(what: str, body: str) -> str:
        nonlocal done
        async with sem:
            out, usage = await complete(C.LLM_FAST_MODEL, [{"role": "user", "content": SUMMARY_PROMPT.format(what=what, body=body)}])
            done += 1
            if progress:
                await progress("summarizing", done, len(groups) + 1, usage)
            return out

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

    usage = {"input": 0, "cache_read": 0, "output": 0}
    enrich_ = enrich_ and C.HAS_LLM and len(chunks) >= C.CONTEXT_MIN_CHUNKS and len(doc) >= C.CONTEXT_MIN_DOC_CHARS
    section_summaries: list[str] = []
    doc_summary = ""
    if enrich_:
        queue: asyncio.Queue = asyncio.Queue()

        async def progress(stage, done, total, u):
            for key in usage:
                usage[key] += u[key]
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
