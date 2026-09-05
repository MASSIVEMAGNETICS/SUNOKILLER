"""Process-isolated execution boundary for capability-leased work."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, Mapping, Optional, Tuple

from .contracts import CapabilityLease, ExecutionReceipt, HMACAuthority, digest_json
from .state import NO_SNAPSHOT_PRECONDITION, SQLiteStateStore


class WorkerExecutionError(RuntimeError):
    pass


class WorkerTimeout(WorkerExecutionError):
    pass


@dataclass(frozen=True)
class WorkerSpec:
    """Execution limits and exact import target.

    Security-sensitive capability/resource policy is deliberately NOT supplied
    here. A caller may request a registered worker and tune bounded execution
    limits, but cannot declare what that worker is authorized to do.
    """

    module: str
    function: str
    timeout_seconds: float = 30.0
    max_memory_mb: Optional[int] = 1024
    max_cpu_seconds: Optional[int] = 30

    @property
    def worker_id(self) -> str:
        return "{}:{}".format(self.module, self.function)


@dataclass(frozen=True)
class WorkerPolicy:
    worker_id: str
    capability: str
    logical_resource: str
    filesystem_fields: Tuple[str, ...] = ()


# This registry is part of the trusted runtime boundary. Callers cannot weaken
# a worker policy by constructing WorkerSpec differently. New workers must be
# intentionally registered here (or, in a later version, through a signed
# manifest registry with equivalent trust semantics).
_TRUSTED_WORKER_POLICIES = {
    "sunokiller.runtime.demo_worker:update_counter": WorkerPolicy(
        worker_id="sunokiller.runtime.demo_worker:update_counter",
        capability="audio.master",
        logical_resource="catalog/masters/demo",
    ),
    "sunokiller.omen:mastering_worker": WorkerPolicy(
        worker_id="sunokiller.omen:mastering_worker",
        capability="audio.master",
        logical_resource="catalog/masters/omen",
        filesystem_fields=("input_path", "output_path"),
    ),
}


def _minimal_environment() -> Dict[str, str]:
    allowed = {}
    for key in (
        "PATH",
        "PYTHONPATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
    ):
        value = os.environ.get(key)
        if value:
            allowed[key] = value
    return allowed


def _resource_limiter(
    max_memory_mb: Optional[int],
    max_cpu_seconds: Optional[int],
):
    if os.name != "posix":
        return None

    def apply_limits() -> None:
        import resource

        if max_memory_mb:
            limit = int(max_memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        if max_cpu_seconds:
            cpu = int(max_cpu_seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))

    return apply_limits


class IsolatedRunner:
    """Run a registered, capability-leased worker in a separate interpreter.

    The child receives only the JSON job payload and a minimized environment.
    It cannot write canonical state directly; it may only propose `_state`,
    which the parent commits transactionally after successful execution.

    Security contract:
    - exact worker code target must exist in the trusted worker-policy registry;
    - the signed lease subject must equal that exact module:function worker_id;
    - capability/logical-resource policy comes only from the trusted registry;
    - required filesystem fields come only from the trusted registry and use
      filesystem-aware path containment, never logical ':' namespace matching;
    - lease validity is rechecked after the subprocess returns;
    - Human STOP/revocation is checked atomically with canonical state commit.
    """

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

    @staticmethod
    def _policy_for(worker: WorkerSpec) -> WorkerPolicy:
        policy = _TRUSTED_WORKER_POLICIES.get(worker.worker_id)
        if policy is None:
            raise WorkerExecutionError(
                "worker is not registered in trusted policy: {}".format(worker.worker_id)
            )
        return policy

    def _verify_worker_and_resources(
        self,
        *,
        lease: CapabilityLease,
        worker: WorkerSpec,
        payload: Mapping[str, Any],
    ) -> WorkerPolicy:
        policy = self._policy_for(worker)
        if lease.subject != policy.worker_id:
            raise WorkerExecutionError(
                "signed lease subject {} does not authorize worker {}".format(
                    lease.subject, policy.worker_id
                )
            )

        revoked_ids = self.state_store.revoked_ids()
        self.authority.verify_lease(
            lease,
            required_capability=policy.capability,
            resource=policy.logical_resource,
            revoked_ids=revoked_ids,
            resource_kind="logical",
        )

        for field in policy.filesystem_fields:
            raw_value = payload.get(field)
            if not isinstance(raw_value, str) or not raw_value:
                raise WorkerExecutionError(
                    "payload resource field {} must be a non-empty path string".format(field)
                )
            actual_resource = str(Path(raw_value).expanduser().resolve())
            self.authority.verify_lease(
                lease,
                required_capability=policy.capability,
                resource=actual_resource,
                revoked_ids=revoked_ids,
                resource_kind="filesystem",
            )
        return policy

    def execute(
        self,
        *,
        lease: CapabilityLease,
        worker: WorkerSpec,
        payload: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], ExecutionReceipt]:
        input_payload = dict(payload)
        policy = self._verify_worker_and_resources(
            lease=lease,
            worker=worker,
            payload=input_payload,
        )

        pre = self.state_store.load_latest(self.state_key)
        pre_hash = pre.state_hash if pre else digest_json({})
        started = int(time.time())
        input_hash = digest_json(input_payload)

        envelope = {
            "module": worker.module,
            "function": worker.function,
            "payload": input_payload,
        }

        try:
            completed = subprocess.run(
                [sys.executable, "-m", "sunokiller.runtime.worker_entry"],
                input=json.dumps(envelope),
                text=True,
                capture_output=True,
                timeout=worker.timeout_seconds,
                env=_minimal_environment(),
                check=False,
                preexec_fn=_resource_limiter(
                    worker.max_memory_mb,
                    worker.max_cpu_seconds,
                ),
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkerTimeout(worker.worker_id) from exc

        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip() or "worker failed"
            raise WorkerExecutionError(error)

        try:
            message = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise WorkerExecutionError("worker returned invalid JSON") from exc

        if not message.get("ok"):
            raise WorkerExecutionError(message.get("error", "unknown worker error"))

        # A long-running worker may cross lease expiry or be revoked while the
        # subprocess is running. Revalidate signed authority immediately before
        # accepting its result. Re-read trusted policy as part of that check.
        policy = self._verify_worker_and_resources(
            lease=lease,
            worker=worker,
            payload=input_payload,
        )

        result = dict(message["result"])
        proposed_state = result.pop("_state", None)
        post_hash = pre_hash
        if proposed_state is not None:
            if not isinstance(proposed_state, dict):
                raise WorkerExecutionError("_state must be a dict")
            snapshot = self.state_store.save_snapshot(
                self.state_key,
                proposed_state,
                expected_hash=pre.state_hash if pre else NO_SNAPSHOT_PRECONDITION,
                lease_id=lease.lease_id,
            )
            post_hash = snapshot.state_hash
        else:
            # For stateless work, define the successful execution boundary as
            # the final durable revocation check before receipt construction.
            self.state_store.assert_lease_active(lease.lease_id)

        finished = int(time.time())
        receipt = ExecutionReceipt(
            execution_id="EXEC-" + uuid.uuid4().hex,
            lease_id=lease.lease_id,
            worker=policy.worker_id,
            capability=policy.capability,
            resource=policy.logical_resource,
            started_at=started,
            finished_at=finished,
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            input_hash=input_hash,
            output_hash=digest_json(result),
            status="SUCCESS",
        )
        return result, self.authority.sign_receipt(receipt)
