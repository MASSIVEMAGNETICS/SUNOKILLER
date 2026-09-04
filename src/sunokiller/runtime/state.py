"""Transactional external state store for replaceable workers/models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
import threading
import time
from typing import Any, Dict, Mapping, Optional, Set

from .contracts import canonical_json, digest_json


class StateConflict(RuntimeError):
    pass


# Optimistic-concurrency sentinel meaning: this write is valid only if the
# state key has never had a snapshot. This is intentionally not a hash value.
NO_SNAPSHOT_PRECONDITION = "__NO_SNAPSHOT__"


@dataclass(frozen=True)
class StateSnapshot:
    key: str
    version: int
    state: Dict[str, Any]
    state_hash: str
    created_at: int


class SQLiteStateStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS state_snapshots (
                key TEXT NOT NULL,
                version INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                state_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (key, version)
            );
            CREATE INDEX IF NOT EXISTS idx_state_latest
                ON state_snapshots(key, version DESC);

            CREATE TABLE IF NOT EXISTS lease_revocations (
                lease_id TEXT PRIMARY KEY,
                revoked_at INTEGER NOT NULL,
                reason TEXT NOT NULL
            );
            """
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def load_latest(self, key: str) -> Optional[StateSnapshot]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT key, version, state_json, state_hash, created_at
                FROM state_snapshots
                WHERE key = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return StateSnapshot(
            key=row[0],
            version=int(row[1]),
            state=json.loads(row[2]),
            state_hash=row[3],
            created_at=int(row[4]),
        )

    def save_snapshot(
        self,
        key: str,
        state: Mapping[str, Any],
        *,
        expected_hash: Optional[str] = None,
    ) -> StateSnapshot:
        payload = dict(state)
        state_json = canonical_json(payload)
        state_hash = digest_json(payload)
        created_at = int(time.time())

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    """
                    SELECT version, state_hash
                    FROM state_snapshots
                    WHERE key = ?
                    ORDER BY version DESC
                    LIMIT 1
                    """,
                    (key,),
                ).fetchone()
                current_version = int(row[0]) if row else 0
                current_hash = row[1] if row else None

                if expected_hash == NO_SNAPSHOT_PRECONDITION:
                    if row is not None:
                        raise StateConflict(
                            "state key {} was expected to have no snapshot, got version {}".format(
                                key, current_version
                            )
                        )
                elif expected_hash is not None and current_hash != expected_hash:
                    raise StateConflict(
                        "state hash mismatch for {}: expected {}, got {}".format(
                            key, expected_hash, current_hash
                        )
                    )

                new_version = current_version + 1
                self._conn.execute(
                    """
                    INSERT INTO state_snapshots(key, version, state_json, state_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (key, new_version, state_json, state_hash, created_at),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

        return StateSnapshot(
            key=key,
            version=new_version,
            state=payload,
            state_hash=state_hash,
            created_at=created_at,
        )

    def revoke_lease(self, lease_id: str, reason: str = "revoked") -> None:
        # Revocations share the same connection and therefore the same lock as
        # state commits. This makes the Human STOP path transaction-safe.
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO lease_revocations(lease_id, revoked_at, reason)
                    VALUES (?, ?, ?)
                    ON CONFLICT(lease_id) DO UPDATE SET
                        revoked_at = excluded.revoked_at,
                        reason = excluded.reason
                    """,
                    (lease_id, int(time.time()), reason),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def is_revoked(self, lease_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM lease_revocations WHERE lease_id = ? LIMIT 1",
                (lease_id,),
            ).fetchone()
        return row is not None

    def revoked_ids(self) -> Set[str]:
        with self._lock:
            rows = self._conn.execute("SELECT lease_id FROM lease_revocations").fetchall()
        return {row[0] for row in rows}
