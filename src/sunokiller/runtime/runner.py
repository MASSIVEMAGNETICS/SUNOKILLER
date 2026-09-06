"""Process-isolated execution boundary for capability-leased work."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, Mapping, Optional, Tuple

from .contracts import (
    CapabilityLease,
    ExecutionReceipt,
    HMACAuthority,
    ResourceDenied,
    digest_json,
)
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
    filesystem_inputs: Tuple[str, ...] = ()
    filesystem_outputs: Tuple[str, ...] = ()

    @property
    def filesystem_fields(self) -> Tuple[str, ...]:
        return self.filesystem_inputs + self.filesystem_outputs


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
        filesystem_inputs=("input_path",),
        filesystem_outputs=("output_path",),
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


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Best-effort hard stop for the worker and every descendant it spawned."""
    if process.poll() is not None:
        return

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=1.0)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
        return

    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP alone does not guarantee descendant teardown.
        # taskkill /T is the Windows process-tree primitive available without
        # adding a pywin32 dependency.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            process.kill()
        return

    process.kill()


def _normalized_absolute_path(raw: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(raw)))
    if not path.is_absolute():
        raise ResourceDenied(raw)
    return path


def _relative_if_beneath(resource: Path, scope: Path) -> Optional[Path]:
    try:
        return resource.relative_to(scope)
    except ValueError:
        return None


def _select_signed_filesystem_scope(
    lease: CapabilityLease,
    raw_resource: str,
) -> Tuple[Path, Path]:
    """Select the most-specific signed absolute scope without resolving symlinks.

    Object-level enforcement happens later through descriptor-based O_NOFOLLOW
    traversal. This lexical selection only decides which signed root applies.
    """
    resource = _normalized_absolute_path(raw_resource)
    candidates = []

    for raw_scope in lease.resource_scopes:
        if raw_scope == "*":
            scope = Path("/")
        else:
            scope_candidate = Path(os.path.expanduser(raw_scope))
            if not scope_candidate.is_absolute():
                continue
            scope = Path(os.path.normpath(str(scope_candidate)))
        relative = _relative_if_beneath(resource, scope)
        if relative is not None:
            candidates.append((len(scope.parts), scope, relative))

    if not candidates:
        raise ResourceDenied(raw_resource)

    _, scope, relative = max(candidates, key=lambda item: item[0])
    return scope, relative


def _require_posix_descriptor_containment() -> None:
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise WorkerExecutionError(
            "secure filesystem workers require POSIX descriptor-based no-follow containment"
        )


def _open_directory_no_symlinks(path: Path) -> int:
    """Open an absolute directory by walking from / with O_NOFOLLOW at each hop."""
    _require_posix_descriptor_containment()
    if not path.is_absolute():
        raise ResourceDenied(str(path))

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            if part in ("", "."):
                continue
            next_fd = os.open(part, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _open_parent_from_scope(
    scope_fd: int,
    relative: Path,
    *,
    create_missing: bool,
) -> Tuple[int, str]:
    parts = relative.parts
    if not parts or parts[-1] in ("", ".", ".."):
        raise ResourceDenied(str(relative))

    current_fd = os.dup(scope_fd)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        for part in parts[:-1]:
            if part in ("", ".", ".."):
                raise ResourceDenied(str(relative))
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create_missing:
                    raise
                os.mkdir(part, mode=0o700, dir_fd=current_fd)
                next_fd = os.open(part, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd, parts[-1]
    except Exception:
        os.close(current_fd)
        raise


def _copy_regular_input_from_scope(scope_fd: int, relative: Path, target: Path) -> None:
    parent_fd, name = _open_parent_from_scope(scope_fd, relative, create_missing=False)
    fd = None
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ResourceDenied(str(relative))
        if info.st_nlink != 1:
            raise ResourceDenied("hard-linked input is not allowed: {}".format(relative))
        with os.fdopen(os.dup(fd), "rb") as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def _atomic_commit_output_to_scope(scope_fd: int, relative: Path, source: Path) -> None:
    if not source.is_file():
        raise WorkerExecutionError(
            "worker completed without required staged output: {}".format(source)
        )

    parent_fd, name = _open_parent_from_scope(scope_fd, relative, create_missing=True)
    temp_name = ".{}.sunokiller-{}".format(name, uuid.uuid4().hex)
    fd = None
    try:
        fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        with source.open("rb") as staged, os.fdopen(os.dup(fd), "wb") as destination:
            shutil.copyfileobj(staged, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        os.close(fd)
        fd = None

        # Both source and destination names are relative to the already-opened
        # authorized parent directory. A symlink swap cannot redirect this.
        os.replace(
            temp_name,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        try:
            os.fsync(parent_fd)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(temp_name, dir_fd=parent_fd)
        except OSError:
            pass
        raise
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


class _FilesystemStager:
    """Broker filesystem access through trusted staging + bound directory fds.

    The worker and FFmpeg never receive caller-controlled authorized paths.
    Inputs are copied from no-follow descriptors into a private staging
    directory. Outputs are produced only in staging and atomically committed
    through a directory fd retained from the signed scope.
    """

    def __init__(
        self,
        *,
        lease: CapabilityLease,
        policy: WorkerPolicy,
        payload: Mapping[str, Any],
    ) -> None:
        self.lease = lease
        self.policy = policy
        self.original_payload = dict(payload)
        self.execution_payload = dict(payload)
        self._temp = None
        self._root = None
        self._scope_fds = []
        self._output_commits = []
        self._public_path_map = {}

    def __enter__(self) -> "_FilesystemStager":
        if not self.policy.filesystem_fields:
            return self

        _require_posix_descriptor_containment()
        self._temp = tempfile.TemporaryDirectory(prefix="sunokiller-runtime-")
        self._root = Path(self._temp.name)

        try:
            for index, field in enumerate(self.policy.filesystem_inputs):
                raw = self.original_payload.get(field)
                if not isinstance(raw, str) or not raw:
                    raise WorkerExecutionError(
                        "payload resource field {} must be a non-empty path string".format(field)
                    )
                scope, relative = _select_signed_filesystem_scope(self.lease, raw)
                scope_fd = _open_directory_no_symlinks(scope)
                self._scope_fds.append(scope_fd)
                suffix = Path(raw).suffix
                staged = self._root / "input-{}{}".format(index, suffix)
                _copy_regular_input_from_scope(scope_fd, relative, staged)
                self.execution_payload[field] = str(staged)
                self._public_path_map[str(staged)] = str(_normalized_absolute_path(raw))

            for index, field in enumerate(self.policy.filesystem_outputs):
                raw = self.original_payload.get(field)
                if not isinstance(raw, str) or not raw:
                    raise WorkerExecutionError(
                        "payload resource field {} must be a non-empty path string".format(field)
                    )
                scope, relative = _select_signed_filesystem_scope(self.lease, raw)
                scope_fd = _open_directory_no_symlinks(scope)
                self._scope_fds.append(scope_fd)
                suffix = Path(raw).suffix
                staged = self._root / "output-{}{}".format(index, suffix)
                self.execution_payload[field] = str(staged)
                self._output_commits.append((scope_fd, relative, staged))
                self._public_path_map[str(staged)] = str(_normalized_absolute_path(raw))
        except OSError as exc:
            self._cleanup()
            raise ResourceDenied("descriptor-safe path authorization failed") from exc
        except Exception:
            self._cleanup()
            raise

        return self

    def commit_outputs(self) -> None:
        if self.original_payload.get("dry_run"):
            return
        for scope_fd, relative, staged in self._output_commits:
            _atomic_commit_output_to_scope(scope_fd, relative, staged)

    def restore_public_paths(self, result: Dict[str, Any]) -> Dict[str, Any]:
        restored = dict(result)
        for key, value in tuple(restored.items()):
            if isinstance(value, str) and value in self._public_path_map:
                restored[key] = self._public_path_map[value]
        return restored

    def _cleanup(self) -> None:
        for fd in self._scope_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._scope_fds.clear()
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self._cleanup()


class IsolatedRunner:
    """Run a registered, capability-leased worker in a separate interpreter.

    The child receives only the JSON job payload and a minimized environment.
    It cannot write canonical state directly; it may only propose `_state`,
    which the parent commits transactionally after successful execution.

    Security contract:
    - exact worker code target must exist in the trusted worker-policy registry;
    - the signed lease subject must equal that exact module:function worker_id;
    - capability/logical-resource policy comes only from the trusted registry;
    - filesystem workers use descriptor-bound, no-follow staging on POSIX;
    - the worker never receives the original authorized filesystem paths;
    - lease validity is rechecked after the subprocess returns;
    - Human STOP/revocation and lease expiry are checked inside state commit;
    - external output publication is linearized against STOP/revocation;
    - worker timeout terminates the entire descendant process tree/group.
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
            _select_signed_filesystem_scope(lease, raw_value)
        return policy

    def _run_worker(
        self,
        *,
        worker: WorkerSpec,
        envelope: Mapping[str, Any],
    ) -> Tuple[int, str, str]:
        args = [sys.executable, "-m", "sunokiller.runtime.worker_entry"]
        kwargs = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "env": _minimal_environment(),
        }

        if os.name == "posix":
            kwargs["start_new_session"] = True
            kwargs["preexec_fn"] = _resource_limiter(
                worker.max_memory_mb,
                worker.max_cpu_seconds,
            )
        elif os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        process = subprocess.Popen(args, **kwargs)
        try:
            stdout, stderr = process.communicate(
                json.dumps(dict(envelope)),
                timeout=worker.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _terminate_process_tree(process)
            try:
                stdout, stderr = process.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            raise WorkerTimeout(worker.worker_id) from exc

        return process.returncode, stdout, stderr

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

        with _FilesystemStager(
            lease=lease,
            policy=policy,
            payload=input_payload,
        ) as stager:
            envelope = {
                "module": worker.module,
                "function": worker.function,
                "payload": stager.execution_payload,
            }

            returncode, stdout, stderr = self._run_worker(
                worker=worker,
                envelope=envelope,
            )

            if returncode != 0:
                error = stderr.strip() or stdout.strip() or "worker failed"
                raise WorkerExecutionError(error)

            try:
                message = json.loads(stdout)
            except json.JSONDecodeError as exc:
                raise WorkerExecutionError("worker returned invalid JSON") from exc

            if not message.get("ok"):
                raise WorkerExecutionError(message.get("error", "unknown worker error"))

            # A long-running worker may cross lease expiry or be revoked while
            # running. Revalidate signed authority before accepting any result
            # or committing any staged output.
            policy = self._verify_worker_and_resources(
                lease=lease,
                worker=worker,
                payload=input_payload,
            )

            result = dict(message["result"])
            proposed_state = result.pop("_state", None)
            output_commit_linearized = False

            # File-producing workers write only into the private staging area.
            # The actual publish step is short and runs while SQLite holds the
            # lease commit guard, so a concurrent Human STOP either commits
            # before publication (and blocks it) or after the authorized file
            # replacement has fully completed.
            if policy.filesystem_outputs and not input_payload.get("dry_run"):
                with self.state_store.lease_commit_guard(
                    lease.lease_id,
                    expires_at=lease.expires_at,
                ):
                    stager.commit_outputs()
                output_commit_linearized = True
            else:
                stager.commit_outputs()

            result = stager.restore_public_paths(result)

            post_hash = pre_hash
            if proposed_state is not None:
                if not isinstance(proposed_state, dict):
                    raise WorkerExecutionError("_state must be a dict")
                snapshot = self.state_store.save_snapshot(
                    self.state_key,
                    proposed_state,
                    expected_hash=pre.state_hash if pre else NO_SNAPSHOT_PRECONDITION,
                    lease_id=lease.lease_id,
                    lease_expires_at=lease.expires_at,
                )
                post_hash = snapshot.state_hash
            elif not output_commit_linearized:
                # Stateless/no-side-effect work still needs one final durable
                # authority check before its successful receipt is issued.
                self.state_store.assert_lease_active(
                    lease.lease_id,
                    expires_at=lease.expires_at,
                )

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
