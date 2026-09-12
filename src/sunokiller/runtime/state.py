"""Transactional external state store for replaceable workers/models."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import signal
import socket
import sqlite3
import threading
import time
from typing import Any, Dict, Iterator, Mapping, Optional, Set

from .contracts import LeaseExpired, LeaseRevoked, canonical_json, digest_json


class StateConflict(RuntimeError):
    pass


class StateDeadlineExceeded(RuntimeError):
    pass


# Optimistic-concurrency sentinel meaning: this write is valid only if the
# state key has never had a snapshot. This is intentionally not a hash value.
NO_SNAPSHOT_PRECONDITION = "__NO_SNAPSHOT__"

_BOUNDED_STATE_RESPONSE_BYTES = 4096


def _bounded_kill_and_reap(pid: int) -> None:
    """Stop a state-read helper without blocking the lease-holding parent."""
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    reap_deadline = time.monotonic() + 0.25
    while time.monotonic() < reap_deadline:
        try:
            completed, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if completed == pid:
            return
        time.sleep(0.01)


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

    @staticmethod
    def _assert_deadline(deadline_monotonic: Optional[float]) -> None:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise StateDeadlineExceeded("state operation exceeded end-to-end deadline")

    @contextmanager
    def _locked_until(
        self,
        deadline_monotonic: Optional[float],
    ) -> Iterator[None]:
        if deadline_monotonic is None:
            self._lock.acquire()
        else:
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0 or not self._lock.acquire(timeout=remaining):
                raise StateDeadlineExceeded(
                    "state operation could not acquire the in-process lock before deadline"
                )
        try:
            self._assert_deadline(deadline_monotonic)
            yield
        finally:
            self._lock.release()

    @contextmanager
    def _immediate_transaction(
        self,
        deadline_monotonic: Optional[float],
    ) -> Iterator[None]:
        previous_timeout = int(self._conn.execute("PRAGMA busy_timeout").fetchone()[0])
        began = False
        if deadline_monotonic is not None:
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise StateDeadlineExceeded(
                    "state transaction deadline expired before lock acquisition"
                )
            self._conn.execute(
                "PRAGMA busy_timeout = {}".format(max(1, int(remaining * 1000)))
            )
        try:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                began = True
            except sqlite3.OperationalError as exc:
                if deadline_monotonic is not None and (
                    time.monotonic() >= deadline_monotonic
                    or "locked" in str(exc).lower()
                    or "busy" in str(exc).lower()
                ):
                    raise StateDeadlineExceeded(
                        "state transaction could not acquire the SQLite write lock before deadline"
                    ) from exc
                raise
            self._assert_deadline(deadline_monotonic)
            yield
            self._assert_deadline(deadline_monotonic)
            self._conn.execute("COMMIT")
            began = False
        except Exception:
            if began:
                self._conn.execute("ROLLBACK")
            raise
        finally:
            self._conn.execute("PRAGMA busy_timeout = {}".format(previous_timeout))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _read_scalar_bounded(
        self,
        query: str,
        parameter: str,
        *,
        deadline_monotonic: float,
    ) -> Any:
        """Run one trusted scalar SELECT in a killable helper process.

        The lease-holding parent never performs deadline-bearing SQLite I/O.
        Only fixed internal SELECT statements call this method, and the child
        response is capped before it crosses the process boundary.
        """
        if os.name != "posix" or not hasattr(os, "fork"):
            raise StateDeadlineExceeded(
                "deadline-bound state reads require POSIX process isolation in v0.1"
            )
        self._assert_deadline(deadline_monotonic)

        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        pid = os.fork()
        if pid == 0:
            parent_sock.close()
            connection = None
            try:
                # Never use the inherited connection or parent RLock after fork.
                connection = sqlite3.connect(
                    self.path,
                    check_same_thread=False,
                    isolation_level=None,
                    timeout=0.0,
                )
                connection.execute("PRAGMA query_only=ON")
                row = connection.execute(query, (parameter,)).fetchone()
                value = None if row is None else row[0]
                encoded = json.dumps(
                    {"value": value},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                if len(encoded) > _BOUNDED_STATE_RESPONSE_BYTES:
                    os._exit(2)
                child_sock.sendall(encoded)
                os._exit(0)
            except BaseException:
                os._exit(1)

        child_sock.close()
        parent_sock.setblocking(False)
        chunks = []
        total = 0
        try:
            while True:
                try:
                    chunk = parent_sock.recv(_BOUNDED_STATE_RESPONSE_BYTES)
                    if chunk:
                        total += len(chunk)
                        if total > _BOUNDED_STATE_RESPONSE_BYTES:
                            _bounded_kill_and_reap(pid)
                            raise StateDeadlineExceeded(
                                "bounded state read exceeded its response budget"
                            )
                        chunks.append(chunk)
                except BlockingIOError:
                    pass

                completed, status = os.waitpid(pid, os.WNOHANG)
                if completed == pid:
                    while True:
                        try:
                            chunk = parent_sock.recv(_BOUNDED_STATE_RESPONSE_BYTES)
                        except BlockingIOError:
                            break
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > _BOUNDED_STATE_RESPONSE_BYTES:
                            raise StateDeadlineExceeded(
                                "bounded state read exceeded its response budget"
                            )
                        chunks.append(chunk)
                    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
                        raise StateDeadlineExceeded(
                            "deadline-bound state read failed closed"
                        )
                    break

                if time.monotonic() >= deadline_monotonic:
                    _bounded_kill_and_reap(pid)
                    raise StateDeadlineExceeded(
                        "state read exceeded end-to-end deadline"
                    )
                time.sleep(0.01)
        finally:
            parent_sock.close()

        self._assert_deadline(deadline_monotonic)
        try:
            message = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StateDeadlineExceeded(
                "bounded state read returned an invalid response"
            ) from exc
        if not isinstance(message, dict) or "value" not in message:
            raise StateDeadlineExceeded("bounded state read returned an invalid response")
        return message["value"]

    def load_latest_hash(
        self,
        key: str,
        *,
        deadline_monotonic: Optional[float] = None,
    ) -> Optional[str]:
        """Read only the bounded-runtime pre-state hash."""
        query = """
            SELECT state_hash
            FROM state_snapshots
            WHERE key = ?
            ORDER BY version DESC
            LIMIT 1
        """
        if deadline_monotonic is not None:
            value = self._read_scalar_bounded(
                query,
                key,
                deadline_monotonic=deadline_monotonic,
            )
        else:
            with self._locked_until(None):
                row = self._conn.execute(query, (key,)).fetchone()
            value = None if row is None else row[0]
        if value is not None and not isinstance(value, str):
            raise StateDeadlineExceeded("bounded state hash read returned an invalid value")
        return value

    def load_latest(
        self,
        key: str,
        *,
        deadline_monotonic: Optional[float] = None,
    ) -> Optional[StateSnapshot]:
        if deadline_monotonic is not None:
            raise StateDeadlineExceeded(
                "deadline-bound full-state loading is disabled; read the state hash instead"
            )
        with self._locked_until(deadline_monotonic):
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

    def _lease_is_revoked_locked(self, lease_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM lease_revocations WHERE lease_id = ? LIMIT 1",
            (lease_id,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _assert_not_expired(expires_at: Optional[int]) -> None:
        if expires_at is not None and int(time.time()) >= int(expires_at):
            raise LeaseExpired("lease expired before commit boundary")

    def _assert_lease_active_locked(
        self,
        lease_id: str,
        *,
        expires_at: Optional[int] = None,
    ) -> None:
        if self._lease_is_revoked_locked(lease_id):
            raise LeaseRevoked("lease has been revoked")
        self._assert_not_expired(expires_at)

    def assert_lease_active(
        self,
        lease_id: str,
        *,
        expires_at: Optional[int] = None,
        deadline_monotonic: Optional[float] = None,
    ) -> None:
        """Fail if revocation/expiry is already durable at this instant."""
        if deadline_monotonic is not None:
            if self.is_revoked(
                lease_id,
                deadline_monotonic=deadline_monotonic,
            ):
                raise LeaseRevoked("lease has been revoked")
            self._assert_deadline(deadline_monotonic)
            self._assert_not_expired(expires_at)
            return
        with self._locked_until(deadline_monotonic):
            self._assert_lease_active_locked(lease_id, expires_at=expires_at)

    @contextmanager
    def lease_commit_guard(
        self,
        lease_id: str,
        *,
        expires_at: Optional[int] = None,
        deadline_monotonic: Optional[float] = None,
    ) -> Iterator[None]:
        """Serialize an administrative side effect against STOP/revocation.

        SQLite COMMIT and ROLLBACK cannot be forcibly bounded in-process.
        Therefore bounded execution contexts are refused before BEGIN; only
        explicit administrative callers without an execution deadline may use
        this legacy guard.
        """
        if deadline_monotonic is not None:
            raise StateDeadlineExceeded(
                "deadline-bound external commits are disabled until transaction "
                "finalization is killably supervised"
            )
        with self._locked_until(None):
            with self._immediate_transaction(deadline_monotonic):
                self._assert_lease_active_locked(lease_id, expires_at=expires_at)
                yield

    def save_snapshot(
        self,
        key: str,
        state: Mapping[str, Any],
        *,
        expected_hash: Optional[str] = None,
        lease_id: Optional[str] = None,
        lease_expires_at: Optional[int] = None,
        deadline_monotonic: Optional[float] = None,
    ) -> StateSnapshot:
        """Commit administrative state with optimistic concurrency.

        Bounded execution contexts pass a monotonic deadline and are refused
        before serialization or BEGIN because SQLite COMMIT/ROLLBACK I/O cannot
        be forcibly interrupted in-process. Legacy administrative calls without
        a deadline retain the serialized lease/expiry checks.
        """
        if deadline_monotonic is not None:
            raise StateDeadlineExceeded(
                "deadline-bound state mutation is disabled until transaction "
                "finalization is killably supervised"
            )
        payload = dict(state)
        state_json = canonical_json(payload)
        state_hash = digest_json(payload)

        with self._locked_until(deadline_monotonic):
            with self._immediate_transaction(deadline_monotonic):
                if lease_id is not None:
                    self._assert_lease_active_locked(
                        lease_id,
                        expires_at=lease_expires_at,
                    )

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
                created_at = int(time.time())
                self._conn.execute(
                    """
                    INSERT INTO state_snapshots(key, version, state_json, state_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (key, new_version, state_json, state_hash, created_at),
                )

        return StateSnapshot(
            key=key,
            version=new_version,
            state=payload,
            state_hash=state_hash,
            created_at=created_at,
        )

    def revoke_lease(self, lease_id: str, reason: str = "revoked") -> None:
        # Revocations share the same connection and therefore the same lock as
        # state commits and guarded external side-effect finalization.
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

    def is_revoked(
        self,
        lease_id: str,
        *,
        deadline_monotonic: Optional[float] = None,
    ) -> bool:
        if deadline_monotonic is not None:
            value = self._read_scalar_bounded(
                "SELECT 1 FROM lease_revocations WHERE lease_id = ? LIMIT 1",
                lease_id,
                deadline_monotonic=deadline_monotonic,
            )
            if value not in (None, 1):
                raise StateDeadlineExceeded(
                    "bounded revocation read returned an invalid value"
                )
            return value == 1
        with self._locked_until(deadline_monotonic):
            return self._lease_is_revoked_locked(lease_id)

    def revoked_ids(
        self,
        *,
        deadline_monotonic: Optional[float] = None,
    ) -> Set[str]:
        if deadline_monotonic is not None:
            raise StateDeadlineExceeded(
                "deadline-bound revocation enumeration is disabled; query one lease instead"
            )
        with self._locked_until(deadline_monotonic):
            rows = self._conn.execute("SELECT lease_id FROM lease_revocations").fetchall()
        return {row[0] for row in rows}
