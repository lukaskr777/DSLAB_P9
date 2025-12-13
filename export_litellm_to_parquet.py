"""
Export LiteLLM conversations to a single Parquet file compatible with the conversation-level embedding script.

Output schema:
- conversation_id : request_id (str)
- messages        : list[dict]

Optional behavior:
- --include-response : append the assistant response as a final message
"""

import argparse
import json
import os
from typing import Any

import pandas as pd
import psycopg


# ---------------- Minimal coercion helpers ----------------

def to_list_of_dicts(messages: Any) -> list[dict] | None:
    """
    Coerce `messages` into list[dict] if possible.

    Accepted inputs:
      - list[dict]              -> unchanged
      - dict                    -> wrapped in list
      - list[str]               -> wrapped as user messages
      - JSON string (list/dict) -> parsed and handled as above
      - plain string            -> single user message

    Otherwise: return None.
    """
    if messages is None:
        return None

    if isinstance(messages, list):
        if not messages:
            return None
        if all(isinstance(x, dict) for x in messages):
            return messages
        if all(isinstance(x, str) for x in messages):
            out = [{"role": "user", "content": s} for s in messages if s.strip()]
            return out or None
        return None

    if isinstance(messages, dict):
        return [messages]

    if isinstance(messages, (bytes, bytearray, memoryview)):
        try:
            messages = bytes(messages).decode("utf-8", errors="replace")
        except Exception:
            return None

    if isinstance(messages, str):
        s = messages.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = None

        if isinstance(parsed, list):
            if parsed and all(isinstance(x, dict) for x in parsed):
                return parsed
            if parsed and all(isinstance(x, str) for x in parsed):
                out = [{"role": "user", "content": t} for t in parsed if t.strip()]
                return out or None
            return None

        if isinstance(parsed, dict):
            return [parsed]

        # fallback: plain text
        return [{"role": "user", "content": s}]

    return None


def response_to_text(response: Any) -> str | None:
    """
    Minimal response extraction.

    - str -> itself
    - JSON string -> parsed; if str -> use it
    - everything else -> ignored
    """
    if response is None:
        return None

    if isinstance(response, str):
        s = response.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
            if isinstance(parsed, str):
                return parsed.strip() or None
            return None
        except Exception:
            return s

    return None


# ---------------- Main export script ----------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export LiteLLM conversations to Parquet for embeddings"
    )
    parser.add_argument("--host", default=os.getenv("DB_HOST"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DB_PORT", "5432")))
    parser.add_argument("--database", default=os.getenv("DB_NAME"))
    parser.add_argument("--user", default=os.getenv("DB_USER"))
    parser.add_argument("--password", default=os.getenv("DB_PASSWORD"))

    parser.add_argument(
        "--out",
        required=True,
        help="Output Parquet path (single file)",
    )
    parser.add_argument(
        "--fetch-size",
        type=int,
        default=5000,
        help="Number of rows fetched per cursor iteration",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional LIMIT for testing",
    )
    parser.add_argument(
        "--include-response",
        action="store_true",
        help="Append assistant response as final message",
    )

    args = parser.parse_args()

    missing = [k for k in ["host", "database", "user", "password"] if not getattr(args, k)]
    if missing:
        raise SystemExit(
            f"Missing DB params: {missing} "
            "(set args or env DB_HOST/DB_NAME/DB_USER/DB_PASSWORD)"
        )

    query = """
    SELECT
        request_id,
        messages,
        response
    FROM "LiteLLM_SpendLogs"
    WHERE messages IS NOT NULL
    ORDER BY "startTime" DESC
    """
    if args.limit:
        query += f" LIMIT {args.limit}"

    conninfo = dict(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=args.password,
    )

    out_rows: list[dict] = []
    total = kept = 0

    with psycopg.connect(**conninfo) as conn:
        with conn.cursor(name="litellm_export_cursor") as cur:
            cur.itersize = args.fetch_size
            cur.execute(query)

            for request_id, messages, response in cur:
                total += 1

                msgs = to_list_of_dicts(messages)
                if msgs is None:
                    continue

                if args.include_response:
                    resp_txt = response_to_text(response)
                    if resp_txt:
                        msgs = msgs + [{"role": "assistant", "content": resp_txt}]

                out_rows.append(
                    {
                        "conversation_id": str(request_id),
                        "messages": msgs,
                    }
                )
                kept += 1

                if total % 50_000 == 0:
                    print(f"[{total:,}] kept={kept:,} ({kept / total:.1%})")

    df = pd.DataFrame(out_rows, columns=["conversation_id", "messages"])
    df.to_parquet(args.out, index=False, engine="pyarrow")

    print(
        f"Done. total={total:,} kept={kept:,} "
        f"wrote={len(df):,} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
