"""One runnable check: `uv run python tests/test_atlas.py`. Exercises the chunker
invariants, RRF, and a real hybrid retrieve round-trip against a throwaway DB."""

import os
import sys
import tempfile
from pathlib import Path

os.environ["ATLAS_DATA"] = tempfile.mkdtemp()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.answer import parse_cites  # noqa: E402
from atlas.ingest import chunk, group_sections  # noqa: E402
from atlas.store import Store, rrf  # noqa: E402

DOC = """# Atlas
Intro paragraph about the Atlas system and its hybrid retrieval design.

## Retrieval
Dense retrieval embeds text with bge-small. BM25 handles exact terms like ERR_QUOTA_7731.
""" + ("Filler sentence number %d keeps this section long enough to force a split. " * 60) % tuple(range(60)) + """

## Storage
LanceDB stores vectors and the full-text index in one table. Zebra-striped rows are not a thing.

```python
# not a heading: comments inside fences stay code
x = 1
```

## Runon
""" + "AlphaBeta GammaDelta " * 200  # no sentence breaks: forces the token-boundary hard split


def test_cites():
    text = "A [1] b [2, 3] c [12] d [4"  # trailing marker is still streaming
    found, end = parse_cites(text, 0)
    assert found == [0, 1, 2, 11] and text[:end].endswith("[12]")
    assert parse_cites("no markers, [not one]", 0) == ([], 0)


def test_rrf():
    fused = rrf(["a", "b", "c"], ["c", "a", "d"], k=60)
    assert list(fused)[:2] == ["a", "c"]  # both appear in both lists; 'a' ranks higher overall
    assert fused["a"] == 1 / 61 + 1 / 62
    assert "d" in fused and "b" in fused


def test_chunker(store):
    chunks = chunk(DOC, store.tokenizer, budget=120, overlap=20)
    ntok = lambda s: len(store.tokenizer.encode(s, add_special_tokens=False).ids)  # noqa: E731
    assert len(chunks) >= 4
    assert all(ntok(c.text) <= 120 for c in chunks), "budget violated"
    assert chunks[0].heading == "Atlas"
    assert any(c.heading == "Atlas › Retrieval" for c in chunks), "breadcrumb missing"
    assert any(c.heading == "Atlas › Storage" for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(DOC[c.start :].startswith(c.text[:20]) for c in chunks), "start offsets wrong"
    groups = group_sections(chunks)
    assert [h for h, _ in groups] == ["Atlas", "Atlas › Retrieval", "Atlas › Storage", "Atlas › Runon"]
    runon = [c for c in chunks if c.heading == "Atlas › Runon"]
    assert len(runon) >= 2 and all(c.text.startswith("AlphaBeta") for c in runon), "hard split must keep original text"


def test_retrieve(store):
    chunks = chunk(DOC, store.tokenizer, budget=120, overlap=20)
    rows = [
        {"id": f"d:{c.ordinal}", "doc_id": "d", "source": "t.md", "ordinal": c.ordinal, "level": 0, "parent": "",
         "heading": c.heading, "context": "", "text": c.text, "embed_text": f"{c.heading}\n{c.text}"}
        for c in chunks
    ]
    storage = next(r for r in rows if "LanceDB" in r["text"])
    edges = [{"src": "lancedb", "rel": "stores", "dst": "vectors", "chunk_id": storage["id"], "doc_id": "d"}]
    store.add(rows, edges)
    assert store.count() == len(rows)
    assert Store().count() == len(rows), "re-opening an existing DB must not fail"

    # Exact-token query: BM25 must win this one, and hybrid must not lose it.
    for mode in ("bm25", "hybrid", "hybrid+rerank", "full"):
        hits, t, meta = store.retrieve("ERR_QUOTA_7731", k=3, mode=mode)
        assert hits and "ERR_QUOTA_7731" in hits[0].text, (mode, [h.text[:40] for h in hits])
        assert "total" in t
    # Semantic query: dense should surface the storage chunk.
    hits, _, _ = store.retrieve("which database keeps the embeddings", k=3, mode="dense")
    assert any("LanceDB" in h.text for h in hits)
    # Graph: query names a known entity → expansion pulls the chunk in.
    hits, t, meta = store.retrieve("tell me about lancedb", k=3, mode="full")
    assert "lancedb" in meta["entities"] and "graph" in t
    assert any("graph" in h.arms for h in hits)

    # Scoping: a filter that excludes the doc returns nothing; one that includes it works.
    assert store.retrieve("ERR_QUOTA_7731", k=3, mode="full", doc_ids=["other"])[0] == []
    assert store.retrieve("ERR_QUOTA_7731", k=3, mode="full", doc_ids=["d"])[0]

    store.delete_doc("d")
    assert store.count() == 0


if __name__ == "__main__":
    test_cites()
    test_rrf()
    s = Store()
    test_chunker(s)
    test_retrieve(s)
    print("ok")
