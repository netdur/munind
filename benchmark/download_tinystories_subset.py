#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Any, Dict, List

import requests


API_URL = "https://datasets-server.huggingface.co/rows"


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> List[Dict[str, Any]]:
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("rows", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TinyStories subset as JSONL")
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument(
        "--output",
        default="benchmark/data/tinystories_subset.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    collected = []
    offset = 0
    retries = 3

    print(f"Downloading subset from {args.dataset} ({args.split})")
    print(f"Target rows: {args.limit}")

    while len(collected) < args.limit:
        length = min(args.batch_size, args.limit - len(collected))

        for attempt in range(1, retries + 1):
            try:
                rows = fetch_rows(args.dataset, args.config, args.split, offset, length)
                break
            except Exception as exc:
                if attempt == retries:
                    raise RuntimeError(f"failed to fetch rows at offset {offset}: {exc}") from exc
                time.sleep(attempt)
        if not rows:
            print("No more rows returned from server.")
            break

        for row_obj in rows:
            row = row_obj.get("row", {})
            text = row.get("text") or row.get("story")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if len(text) < args.min_chars:
                continue
            collected.append(
                {
                    "id": row_obj.get("row_idx", offset),
                    "text": text,
                }
            )
            if len(collected) >= args.limit:
                break

        offset += len(rows)
        print(f"Collected {len(collected)}/{args.limit} rows...")

    with open(args.output, "w", encoding="utf-8") as f:
        for row in collected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(collected)} rows to {args.output}")


if __name__ == "__main__":
    main()
