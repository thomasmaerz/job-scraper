"""Continuous Freehire compatibility worker."""

import logging
import os

import backfill_freehire_compat


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        limit = int(os.getenv("FREEHIRE_CLASSIFY_LIMIT", "200"))
    except ValueError as exc:
        raise ValueError("FREEHIRE_CLASSIFY_LIMIT must be a positive integer") from exc
    if limit <= 0:
        raise ValueError("FREEHIRE_CLASSIFY_LIMIT must be a positive integer")
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
