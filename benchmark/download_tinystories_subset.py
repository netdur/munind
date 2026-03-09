#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import requests


API_URL = "https://datasets-server.huggingface.co/rows"


class RetryableFetchError(Exception):
    def __init__(self, message: str, retry_after_seconds: Optional[float] = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    timeout_seconds: int,
) -> List[Dict[str, Any]]:
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }

    try:
        resp = requests.get(API_URL, params=params, timeout=timeout_seconds)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 429 or (status_code is not None and status_code >= 500):
            retry_after = None
            if exc.response is not None:
                retry_after_header = exc.response.headers.get("Retry-After")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        retry_after = None
            raise RetryableFetchError(
                f"HTTP {status_code} at offset {offset}", retry_after_seconds=retry_after
            ) from exc
        raise
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        raise RetryableFetchError(f"network error at offset {offset}: {exc}") from exc

    payload = resp.json()
    return payload.get("rows", [])


def compute_backoff_seconds(
    attempt: int,
    base_seconds: float,
    max_seconds: float,
    retry_after_seconds: Optional[float],
) -> float:
    if retry_after_seconds is not None and retry_after_seconds > 0:
        return min(max_seconds, retry_after_seconds)

    exponential = base_seconds * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, base_seconds)
    return min(max_seconds, exponential + jitter)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TinyStories subset as JSONL")
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--retry-backoff-base", type=float, default=1.0)
    parser.add_argument("--retry-backoff-max", type=float, default=60.0)
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.25,
        help="Fixed delay between successful API calls to reduce rate-limit pressure",
    )
    parser.add_argument(
        "--output",
        default="benchmark/data/tinystories_subset.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    collected = []
    offset = 0

    print(f"Downloading subset from {args.dataset} ({args.split})")
    print(f"Target rows: {args.limit}")

    while len(collected) < args.limit:
        length = min(args.batch_size, args.limit - len(collected))

        rows: List[Dict[str, Any]] = []
        for attempt in range(1, args.max_retries + 1):
            try:
                rows = fetch_rows(
                    args.dataset,
                    args.config,
                    args.split,
                    offset,
                    length,
                    args.request_timeout,
                )
                break
            except RetryableFetchError as exc:
                if attempt == args.max_retries:
                    raise RuntimeError(
                        f"failed to fetch rows at offset {offset} after {args.max_retries} attempts: {exc}"
                    ) from exc

                backoff = compute_backoff_seconds(
                    attempt,
                    args.retry_backoff_base,
                    args.retry_backoff_max,
                    exc.retry_after_seconds,
                )
                print(
                    f"Retryable fetch error at offset {offset} (attempt {attempt}/{args.max_retries}): {exc}. "
                    f"Sleeping {backoff:.2f}s"
                )
                time.sleep(backoff)
            except Exception as exc:
                raise RuntimeError(f"failed to fetch rows at offset {offset}: {exc}") from exc

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

        if args.sleep_between_requests > 0:
            time.sleep(args.sleep_between_requests)

    with open(args.output, "w", encoding="utf-8") as f:
        for row in collected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(collected)} rows to {args.output}")


if __name__ == "__main__":
    main()
