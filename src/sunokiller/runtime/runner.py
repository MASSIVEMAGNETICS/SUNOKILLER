"""Process-isolated execution boundary for capability-leased work."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib
import multiprocessing as mp
import os
import time
from typing import Any, Dict, Mapping, Optional, Tuple

from .contracts import CapabilityLease, ExecutionReceipt, HMACAuthority, digest_json
from .state import SQLiteStateStore


class WorkerExecutionError(RuntimeError):
    pass


class WorkerTimeout(WorkerExecutionError):
    pass


@dataclass(frozen=True)
class WorkerSpec:
    module: str
    function: str
    capability: str
    resource: str
    timeout_seconds: float = 30.0
    max_memory_mb: Optional[int] = 1024
    max_cpu_seconds: Optional[int] = 30

    @property
    def worker_id(self) -> str:
        return "{}:{}".format(self.module, self.function)


def _apply_resource_limits(
    max_memory_mb: Optional[int],
    max_cpu_seconds: Optional[int],
) -> None:
    try:
        import resource  # POSIX only
    except ImportError:
        return

    if max_memory_mb:
        limit = int(max_memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    if max_cpu_seconds:
        cpu = int(max_cpu_seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))


def _minimize_environment() -> None:
    allowed = {}
    for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            allowed[key] = value
    os.environ.clear()
    os.environ.update(allowed)


def _worker_entry(
    conn: Any,
    module: str,
    function: str,
    payload: Dict[str, Any],
    max_memory_mb: Optional[int],
    max_cpu_seconds: Optional[int],
) -> None:
    try:
        _apply_resource_limits(max_memory_mb, max_cpu_seconds)
        _minimize_environment()
        target = getattr(importlib.import_module(module), function)
        result = target(payload)
        if not isinstance(result, dict):
            raise TypeError("worker must return a dict")
        conn.send({"ok": True, "result": result})
    except BaseException as exc:
        conn.send({"ok": False, "error": "{}: {}".format(type(exc).__name__, exc)})
    finally:
        conn.close()


class IsolatedRunner:
    """Run an authorized worker in a child process with external state commit."""

    def __init__(
        self,
        *,
        authority: HMACAuthority,
        state_store: SQLiteStateStore,
        state_key: str,
    ) -> None:
        self.authority = authority
        self.state_store = state_store
        self.state_key = state_key

    def execute(
        self,
        *,
        lease: CapabilityLease,
        worker: WorkerSpec,
        payload: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], ExecutionReceipt]:
        self.authority.verify_lease(
            lease,
            required_capability=worker.capability,
            resource=worker.resource,
            revoked_ids=self.state_store.revoked_ids(),
        )

        pre = self.state_store.load_latest(self.state_key)
        pre_hash = pre.state_hash if pre else digest_json({})
        started = int(time.time())
        input_hash = digest_json(dict(payload))

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=_worker_entry,
            args=(
                child_conn,
                worker.module,
                worker.function,
                dict(payload),
                worker.max_memory_mb,
                worker.max_cpu_seconds,
            ),
        )
        process.start()
        child_conn.close()

        message = None
        try:
            if not parent_conn.poll(worker.timeout_seconds):
                process.terminate()
                process.join(timeout=2.0)
                raise WorkerTimeout(worker.worker_id)
            message = parent_conn.recv()
        finally:
            parent_conn.close()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)

        if not message or not message.get("ok"):
            error = message.get("error", "unknown worker error") if message else "no worker response"
            raise WorkerExecutionError(error)

        result = dict(message["result"])
        proposed_state = result.pop("_state", None)
        post_hash = pre_hash
        if proposed_state is not None:
            if not isinstance(proposed_state, dict):
                raise WorkerExecutionError("_state must be a dict")
            snapshot = self.state_store.save_snapshot(
                self.state_key,
                proposed_state,
                expected_hash=pre.state_hash if pre else None,
            )
            post_hash = snapshot.state_hash

        finished = int(time.time())
        receipt_seed = "{}:{}:{}:{}".format(
            lease.lease_id, started, worker.worker_id, input_hash
        )
        receipt = ExecutionReceipt(
            execution_id="EXEC-" + sha256(receipt_seed.encode("utf-8")).hexdigest()[:24],
            lease_id=lease.lease_id,
            worker=worker.worker_id,
            capability=worker.capability,
            resource=worker.resource,
            started_at=started,
            finished_at=finished,
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            input_hash=input_hash,
            output_hash=digest_json(result),
            status="SUCCESS",
        )
        signed = self.authority.sign_receipt(receipt)
        return result, signed
