from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import supabase_utils


class LinkedInCircuitOpen(RuntimeError):
    pass


class LinkedInGrantRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class ConsumedGrant:
    grant_id: str
    started_at: datetime


class DurableLinkedInRequestGate:
    def __init__(self, producer: str, *, db: Any = None) -> None:
        self.producer = producer
        self.db = db

    def acquire(self, request_kind: str, request_key: str) -> ConsumedGrant:
        while True:
            grant = supabase_utils.acquire_linkedin_request_grant(
                self.producer,
                request_kind,
                request_key,
                db=self.db,
            )
            outcome = grant.get("outcome")
            if outcome == "circuit_open":
                raise LinkedInCircuitOpen(str(grant.get("reason") or "LinkedIn circuit is open"))
            if outcome == "wait":
                wait_ms = grant.get("wait_ms")
                if not isinstance(wait_ms, int) or wait_ms < 0:
                    raise LinkedInGrantRejected("request grant returned an invalid wait")
                time.sleep(min(wait_ms / 1000, 60))
                continue
            if outcome != "grant" or not grant.get("grant_id"):
                raise LinkedInGrantRejected("request grant was rejected")
            consumed = supabase_utils.consume_linkedin_request_grant(
                str(grant["grant_id"]), self.producer, db=self.db
            )
            if not consumed.get("consumed"):
                if consumed.get("reason") == "circuit_open":
                    raise LinkedInCircuitOpen("LinkedIn circuit opened before request start")
                continue
            started_at = consumed.get("started_at")
            if not isinstance(started_at, str):
                raise LinkedInGrantRejected("consumed grant omitted started_at")
            return ConsumedGrant(
                grant_id=str(grant["grant_id"]),
                started_at=datetime.fromisoformat(started_at.replace("Z", "+00:00")),
            )

    def finish(self, grant: ConsumedGrant, response_class: str, http_status: int | None) -> None:
        finished = supabase_utils.finish_linkedin_request_grant(
            grant.grant_id,
            self.producer,
            response_class,
            http_status,
            db=self.db,
        )
        if not finished:
            raise LinkedInGrantRejected("request grant was invalidated before completion")

    def open_circuit(
        self,
        grant: ConsumedGrant,
        reason: str,
        http_status: int | None,
    ) -> None:
        opened = supabase_utils.open_linkedin_source_circuit(
            grant.grant_id,
            self.producer,
            reason,
            http_status,
            db=self.db,
        )
        if not opened:
            raise LinkedInGrantRejected("request grant was invalidated before circuit open")
