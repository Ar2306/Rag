"""Grounded answer generation with numbered citations.

Retrieved chunks are presented as numbered sources; the model cites `[n]`
after each claim. Markers are parsed out of the stream as they complete and
emitted as `cite` events that map 1:1 onto the hit list.
"""

from __future__ import annotations

import re
from typing import AsyncIterator

from atlas import config as C
from atlas.llm import client, complete, usage_of
from atlas.store import Hit

SYSTEM = (
    "You are Atlas, a retrieval-grounded assistant.\n\n"
    "Answer the user's question using ONLY the numbered sources. After every factual claim, cite the "
    "source number in square brackets, e.g. [2] or [1][3]. If the sources do not contain the answer, "
    "say so plainly and do not guess. Be direct and concise; use markdown headings, lists and tables "
    "only when they aid clarity. Do not mention \"the sources provided\" — just answer and cite."
)

REWRITE = (
    "Rewrite the final user message as ONE standalone question that is fully understandable without the "
    "conversation. Resolve pronouns and references ('it', 'the second one', 'that incident') using the "
    "conversation; preserve every specific term, name and number; add nothing new. If the message already "
    "stands alone, return it unchanged. Return only the question."
)

CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_cites(text: str, start: int = 0) -> tuple[list[int], int]:
    """0-based source indexes of every complete `[n]` / `[n, m]` marker at or after
    `start`, plus the offset to resume scanning from (a half-streamed `[1` is left alone)."""
    found, end = [], start
    for m in CITE.finditer(text, start):
        found += [int(n) - 1 for n in m.group(1).split(",")]
        end = m.end()
    return found, end


async def standalone_question(question: str, history: list[dict]) -> str:
    """Coreference resolution for follow-ups: 'what caused it?' retrieves nothing
    on its own. This is *not* query expansion (which the literature shows hurts) —
    the rewrite must add no information beyond what the conversation contains."""
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:1500]}" for m in history)
    out, _ = await complete(
        C.LLM_FAST_MODEL,
        [
            {"role": "system", "content": REWRITE},
            {"role": "user", "content": f"<conversation>\n{convo}\n</conversation>\n\nFinal user message: {question}"},
        ],
    )
    return out or question


def _sources(hits: list[Hit]) -> str:
    out = []
    for i, h in enumerate(hits, 1):
        title = h.source + (f" › {h.heading}" if h.heading else "")
        ctx = f"({h.context})\n" if h.context else ""
        out.append(f"[{i}] {title}\n{ctx}{h.text}")
    return "\n\n".join(out)


async def stream_answer(question: str, hits: list[Hit], history: list[dict]) -> AsyncIterator[dict]:
    """Yields {"t": "block"} | {"t": "text", "text"} | {"t": "cite", "doc"} | {"t": "done", ...}."""
    messages = [
        {"role": "system", "content": SYSTEM},
        *history,
        {"role": "user", "content": f"<sources>\n{_sources(hits)}\n</sources>\n\nQuestion: {question}"},
    ]
    stream = await client().chat.completions.create(
        model=C.LLM_MODEL, messages=messages, stream=True, stream_options={"include_usage": True}
    )
    yield {"t": "block"}
    text, scanned, cited, usage, stop = "", 0, set(), None, None
    async for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        stop = choice.finish_reason or stop
        delta = choice.delta.content or ""
        if not delta:
            continue
        text += delta
        yield {"t": "text", "text": delta}
        found, scanned = parse_cites(text, scanned)
        for n in found:
            if 0 <= n < len(hits) and n not in cited:
                cited.add(n)
                yield {"t": "cite", "doc": n}
    yield {"t": "done", "stop": stop or "stop", "usage": usage_of(usage), "answer": text}
