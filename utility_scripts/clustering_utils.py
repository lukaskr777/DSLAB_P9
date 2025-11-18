from typing import Any, Iterable
import json

import numpy as np


def ensure_list_of_dicts(obj: Any) -> list[dict[str, Any]]:
    """Best-effort conversion of `obj` to a list of message dicts."""
    # numpy array of dicts
    if isinstance(obj, np.ndarray):
        return [m for m in obj.tolist() if isinstance(m, dict)]

    # plain list
    if isinstance(obj, list):
        return [m for m in obj if isinstance(m, dict)]

    # single dict
    if isinstance(obj, dict):
        return [obj]

    # JSON string (if ever needed)
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return ensure_list_of_dicts(parsed)
        except Exception:
            return []

    # generic iterable fallback
    try:
        it = list(obj)
        return [m for m in it if isinstance(m, dict)]
    except Exception:
        return []


def extract_text_from_content(content: Any) -> str:
    """
    Extract human-readable text from the nested `content` structure.

    Priority:
    - content["text"] if non-empty
    - join all parts[i]["text"] in content["parts"]
    - join all blocks[i]["text"] in content["blocks"]
    """
    if content is None:
        return ""

    if not isinstance(content, dict):
        return str(content)

    texts: list[str] = []

    # 1) direct text field
    txt = content.get("text")
    if isinstance(txt, str) and txt.strip():
        texts.append(txt.strip())

    # 2) parts -> [{'text': ..., 'type': ...}, ...]
    parts = content.get("parts")
    if parts is not None:
        try:
            parts_iter = list(parts)
        except TypeError:
            parts_iter = [parts]
        for p in parts_iter:
            if isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())

    # 3) blocks -> [{'text': ..., 'type': 'response', ...}, ...]
    blocks = content.get("blocks")
    if blocks is not None:
        try:
            blocks_iter = list(blocks)
        except TypeError:
            blocks_iter = [blocks]
        for b in blocks_iter:
            if isinstance(b, dict):
                t = b.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())

    return "\n".join(texts)


def flatten_messages_to_text(messages: Iterable[dict[str, Any]]) -> str:
    """Join a list of message dicts into a single conversation string."""
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip()
        content_text = extract_text_from_content(m.get("content"))

        if not content_text:
            continue

        if role:
            parts.append(f"{role}: {content_text}")
        else:
            parts.append(content_text)

    return "\n".join(parts)
