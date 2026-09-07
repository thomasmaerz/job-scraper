"""Incremental Freehire compatibility worker."""

import logging
import os

import backfill_freehire_compat


def classify_page_size_from_env() -> int:
    try:
        page_size = int(os.getenv("FREEHIRE_CLASSIFY_PAGE_SIZE", "500"))
    except ValueError as exc:
        raise ValueError("FREEHIRE_CLASSIFY_PAGE_SIZE must be a positive integer") from exc
    if page_size <= 0:
        raise ValueError("FREEHIRE_CLASSIFY_PAGE_SIZE must be a positive integer")
    return page_size


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    page_size = classify_page_size_from_env()
    result = backfill_freehire_compat.run(
        apply=True,
        limit=page_size,
        drain_backlog=True,
    )
    logging.info(
        "Freehire compatibility status=%s stats=%s",
        backfill_freehire_compat.result_status(result),
        result,
    )
    print(result)
    raise SystemExit(backfill_freehire_compat.result_exit_code(result))
