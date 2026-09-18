"""Grounded answer generation with the API's native citations.

Each retrieved chunk is passed as its own custom-content document, so every
citation Claude returns carries a `document_index` that maps 1:1 onto our hit
list — no regex over "[3]" markers, no hallucinated references.
"""

from __future__ import annotations

from typing import AsyncIterator

import anthropic

from atlas import config as C
from atlas.store import Hit

SYSTEM = """You are Atlas, a retrieval-grounded assistant.

Answer the user's question using ONLY the provided documents. Cite the documents for every factual claim. If the documents do not contain the answer, say so plainly and do not guess. Be direct and concise; use markdown headings, lists and tables only when they aid clarity. Do not mention "the documents provided" — just answer and cite."""


REWRITE = (
    "Rewrite the final user message as ONE standalone question that is fully understandable without the "
    "conversation. Resolve pronouns and references ('it', 'the second one', 'that incident') using the "
    "conversation; preserve every specific term, name and number; add nothing new. If the message already "
    "stands alone, return it unchanged. Return only the question."
)


async def standalone_question(question: str, history: list[dict]) -> str:
    """Coreference resolution for follow-ups: 'what caused it?' retrieves nothing
    on its own. This is *not* query expansion (which the literature shows hurts) —
    the rewrite must add no information beyond what the conversation contains."""
    if not history:
        return question
    client = anthropic.AsyncAnthropic()
    convo = "\n".join(f"{m['role']}: {m['content'][:1500]}" for m in history)
    resp = await client.messages.create(
        model=C.CONTEXT_MODEL,
        max_tokens=200,
        system=REWRITE,
        messages=[{"role": "user", "content": f"<conversation>\n{convo}\n</conversation>\n\nFinal user message: {question}"}],
    )
    out = next((b.text for b in resp.content if b.type == "text"), "").strip()
    return out or question


def _documents(hits: list[Hit]) -> list[dict]:
    return [
        {
            "type": "document",
            "source": {"type": "content", "content": [{"type": "text", "text": h.text}]},
            "title": f"{h.source}" + (f" › {h.heading}" if h.heading else ""),
            # `context` is shown to the model but never cited from — the ideal slot
            # for the contextual-retrieval sentence.
            **({"context": h.context} if h.context else {}),
            "citations": {"enabled": True},
        }
        for h in hits
    ]


async def stream_answer(question: str, hits: list[Hit], history: list[dict]) -> AsyncIterator[dict]:
    """Yields {"t": "block"} | {"t": "text", "text"} | {"t": "cite", "doc", "quote"} | {"t": "done", ...}."""
    client = anthropic.AsyncAnthropic()
    messages = [*history, {"role": "user", "content": [*_documents(hits), {"type": "text", "text": question}]}]

    async with client.messages.stream(
        model=C.ANSWER_MODEL,
        max_tokens=4096,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
        output_config={"effort": C.ANSWER_EFFORT},
    ) as stream:
        async for event in stream:
            if event.type == "content_block_start" and event.content_block.type == "text":
                yield {"t": "block"}
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    yield {"t": "text", "text": event.delta.text}
                elif event.delta.type == "citations_delta":
                    c = event.delta.citation
                    yield {"t": "cite", "doc": c.document_index, "quote": c.cited_text}
        final = await stream.get_final_message()
        yield {
            "t": "done",
            "stop": final.stop_reason,
            "usage": {
                "input": final.usage.input_tokens,
                "cache_read": final.usage.cache_read_input_tokens or 0,
                "output": final.usage.output_tokens,
            },
            "answer": "".join(b.text for b in final.content if b.type == "text"),
        }
