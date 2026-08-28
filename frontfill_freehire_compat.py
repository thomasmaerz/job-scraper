"""Continuous Freehire compatibility worker."""

import logging
import os

import backfill_freehire_compat


def classify_limit_from_env() -> int:
    try:
        limit = int(os.getenv("FREEHIRE_CLASSIFY_LIMIT", "300"))
    except ValueError as exc:
        raise ValueError("FREEHIRE_CLASSIFY_LIMIT must be a positive integer") from exc
    if limit <= 0:
        raise ValueError("FREEHIRE_CLASSIFY_LIMIT must be a positive integer")
    if limit > 300:
        raise ValueError("FREEHIRE_CLASSIFY_LIMIT cannot exceed the hourly hard cap of 300")
    return limit


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    limit = classify_limit_from_env()
    result = backfill_freehire_compat.run(
        apply=True,
        limit=limit,
        drain_backlog=os.getenv("FREEHIRE_DRAIN_BACKLOG", "false").lower() == "true",
    )
    logging.info(
        "Freehire compatibility status=%s stats=%s",
        backfill_freehire_compat.result_status(result),
        result,
    )
    print(result)
    raise SystemExit(backfill_freehire_compat.result_exit_code(result))
