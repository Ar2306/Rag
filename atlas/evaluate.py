"""Retrieval eval: proves the pipeline is better than its parts on *your* corpus.

  uv run python -m atlas.evaluate build 40   # synthesise 40 gold (question → chunk) pairs
  uv run python -m atlas.evaluate run        # score every retrieval mode

Gold questions are written by Haiku from randomly sampled leaf chunks, with the
instruction that the question must be answerable from that chunk alone. Metrics
are standard IR: Recall@k, MRR@10, nDCG@10, plus p50 latency per mode.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys

import anthropic

from atlas import config as C
from atlas.store import MODES, Store

GOLD = C.DATA / "gold.json"
QUESTION_PROMPT = (
    "Write ONE specific question that a reader might ask which can be answered using ONLY the passage "
    "below. The question must not be answerable from general knowledge, must not quote the passage "
    "verbatim, and must read naturally (as a user would type it into a search box). "
    "Answer with the question only.\n\n<passage>\n{text}\n</passage>"
)


def build(store: Store, n: int) -> None:
    rows = store.table.search().where("level = 0").select(["id", "text"]).limit(1_000_000).to_list()
    rows = [r for r in rows if len(r["text"]) > 300]  # tiny chunks make degenerate questions
    sample = random.sample(rows, min(n, len(rows)))
    client = anthropic.Anthropic()
    gold = []
    for i, r in enumerate(sample, 1):
        resp = client.messages.create(
            model=C.CONTEXT_MODEL, max_tokens=120,
            messages=[{"role": "user", "content": QUESTION_PROMPT.format(text=r["text"])}],
        )
        q = next(b.text for b in resp.content if b.type == "text").strip()
        gold.append({"question": q, "chunk_id": r["id"]})
        print(f"[{i}/{len(sample)}] {q}")
    GOLD.write_text(json.dumps(gold, indent=1))
    print(f"wrote {len(gold)} → {GOLD}")


def run(store: Store, k: int = 10) -> None:
    gold = json.loads(GOLD.read_text())
    print(f"{len(gold)} questions · k={k}\n")
    print(f"{'mode':16} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR@10':>7} {'nDCG@10':>8} {'p50 ms':>7}")
    for mode in MODES:
        r1 = r5 = r10 = mrr = ndcg = 0.0
        lat = []
        for g in gold:
            hits, t, _ = store.retrieve(g["question"], k=k, mode=mode)
            lat.append(t.get("total", 0))
            ids = [h.id for h in hits]
            if g["chunk_id"] in ids:
                rank = ids.index(g["chunk_id"])
                r1 += rank < 1
                r5 += rank < 5
                r10 += rank < 10
                mrr += 1 / (rank + 1)
                ndcg += 1 / math.log2(rank + 2)  # single relevant doc ⇒ IDCG = 1
        n = len(gold)
        print(f"{mode:16} {r1/n:6.3f} {r5/n:6.3f} {r10/n:6.3f} {mrr/n:7.3f} {ndcg/n:8.3f} {statistics.median(lat):7.1f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    s = Store()
    if cmd == "build":
        build(s, int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    else:
        run(s)
