from typing import Any, Iterable, Mapping
import json

import numpy as np


def ensure_list_of_dicts(obj: Any) -> list[dict[str, Any]]:
    """Best-effort conversion of `obj` to a list of dicts."""
    # Numpy array of dicts
    if isinstance(obj, np.ndarray):
        if obj.dtype != object:
            return []
        return [m for m in obj.tolist() if isinstance(m, dict)]

    # Plain list
    if isinstance(obj, list):
        return [m for m in obj if isinstance(m, dict)]

    # Single dict
    if isinstance(obj, dict):
        return [obj]

    # JSON string
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
        except json.JSONDecodeError:
            return []
        return ensure_list_of_dicts(parsed)

    # Generic iterable fallback
    try:
        it = list(obj)
    except TypeError:
        return []
    return [m for m in it if isinstance(m, dict)]


def _as_iter(obj: Any) -> list[Any]:
    if obj is None:
        return []
    try:
        return list(obj)
    except TypeError:
        return [obj]


def extract_text_from_content(content: Any) -> str:
    """
    Extract human-readable text from a nested `content` structure.

    Collects text from, in order:
    - content["text"]
    - content["parts"][i]["text"]
    - content["blocks"][i]["text"]
    """
    if content is None:
        return ""

    if not isinstance(content, Mapping):
        return str(content)

    texts: list[str] = []

    txt = content.get("text")
    if isinstance(txt, str):
        txt = txt.strip()
        if txt:
            texts.append(txt)

    for p in _as_iter(content.get("parts")):
        if isinstance(p, Mapping):
            t = p.get("text")
            if isinstance(t, str):
                t = t.strip()
                if t:
                    texts.append(t)

    for b in _as_iter(content.get("blocks")):
        if isinstance(b, Mapping):
            t = b.get("text")
            if isinstance(t, str):
                t = t.strip()
                if t:
                    texts.append(t)

    return "\n".join(texts)


def flatten_messages_to_text(messages: Iterable[Mapping[str, Any]]) -> str:
    """Join a sequence of message dicts into a single conversation string."""
    parts: list[str] = []

    for m in messages:
        if not isinstance(m, Mapping):
            continue

        role = str(m.get("role", "")).strip()
        content_text = extract_text_from_content(m.get("content"))
        content_text = content_text.strip()

        if not content_text:
            continue

        if role:
            parts.append(f"{role}: {content_text}")
        else:
            parts.append(content_text)

    return "\n".join(parts)
