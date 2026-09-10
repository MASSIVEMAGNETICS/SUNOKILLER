"""Process-isolated execution boundary for capability-leased work."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
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
    """Caller request for a registered worker and stricter execution limits.

    Capability/resource policy and the maximum resource budget are deliberately
    NOT supplied here. A caller may request a registered worker and choose a
    *stricter* finite budget, but cannot disable or raise the trusted maxima.
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
    max_timeout_seconds: float
    max_memory_mb: int
    max_cpu_seconds: int
    filesystem_inputs: Tuple[str, ...] = ()
    filesystem_outputs: Tuple[str, ...] = ()
    max_input_bytes: int = 0
    max_output_bytes: int = 0

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
        max_timeout_seconds=30.0,
        max_memory_mb=1024,
        max_cpu_seconds=30,
    ),
    "sunokiller.omen:mastering_worker": WorkerPolicy(
        worker_id="sunokiller.omen:mastering_worker",
        capability="audio.master",
        logical_resource="catalog/masters/omen",
        max_timeout_seconds=900.0,
        max_memory_mb=4096,
        max_cpu_seconds=900,
        filesystem_inputs=("input_path",),
        filesystem_outputs=("output_path",),
        max_input_bytes=8 * 1024 * 1024 * 1024,
        max_output_bytes=8 * 1024 * 1024 * 1024,
    ),
}


# Pin child imports to the same installed/editable source root that supplied
# this trusted runner module. `python -I` removes the ambient CWD, PYTHONPATH,
# user-site and other Python environment influence; this path is then inserted
# explicitly inside the fresh interpreter before worker_entry is imported.
_TRUSTED_SOURCE_ROOT = str(Path(__file__).resolve().parents[2])


def _minimal_environment() -> Dict[str, str]:
    """Return only non-Python ambient variables needed by bounded workers.

    PYTHONPATH is intentionally never forwarded. The child is also launched
    with `-I`, which ignores Python-specific environment variables and removes
    the current working directory from the import search path.
    """
    allowed: Dict[str, str] = {}
    for key in (
        "PATH",
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


def _isolated_worker_command() -> Tuple[str, ...]:
    bootstrap = (
        "import runpy,sys;"
        "sys.path.insert(0,{!r});"
        "runpy.run_module('sunokiller.runtime.worker_entry',run_name='__main__')"
    ).format(_TRUSTED_SOURCE_ROOT)
    return (sys.executable, "-I", "-c", bootstrap)


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Best-effort hard stop for the worker and every descendant it spawned."""
    if os.name == "posix":
        # The session/process-group can outlive its leader. Always target the
        # group even when poll() says the Python worker has already exited.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        return

    if process.poll() is not None:
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


def _bounded_kill_and_reap(pid: int) -> None:
    """Kill a helper without ever turning cleanup into an unbounded wait."""
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


def _wait_bounded_io_child(
    pid: int,
    *,
    deadline: float,
    operation: str,
    committed_fd: Optional[int] = None,
) -> None:
    """Wait for a killable I/O child, optionally honoring its visibility commit."""
    while True:
        if committed_fd is not None:
            try:
                if os.read(committed_fd, 1) == b"C":
                    _bounded_kill_and_reap(pid)
                    return
            except BlockingIOError:
                pass
        completed, status = os.waitpid(pid, os.WNOHANG)
        if completed == pid:
            if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
                return
            if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 2:
                raise ResourceDenied("{} exceeded its trusted byte limit".format(operation))
            if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 3:
                raise ResourceDenied("{} failed descriptor containment checks".format(operation))
            raise WorkerExecutionError("{} failed in bounded staging child".format(operation))
        if time.monotonic() >= deadline:
            _bounded_kill_and_reap(pid)
            raise WorkerTimeout("{} exceeded end-to-end deadline".format(operation))
        time.sleep(0.01)


def _copy_regular_input_from_scope(
    scope_fd: int,
    relative: Path,
    target: Path,
    *,
    max_bytes: int,
    deadline: float,
) -> None:
    pid = os.fork()
    if pid == 0:
        parent_fd = None
        fd = None
        try:
            parent_fd, name = _open_parent_from_scope(scope_fd, relative, create_missing=False)
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                os._exit(3)
            if max_bytes <= 0 or info.st_size > max_bytes:
                os._exit(2)
            copied = 0
            with os.fdopen(os.dup(fd), "rb") as source, target.open("wb") as destination:
                while True:
                    chunk = source.read(min(1024 * 1024, max_bytes - copied + 1))
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > max_bytes:
                        os._exit(2)
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            os._exit(0)
        except BaseException:
            os._exit(1)
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if parent_fd is not None:
                try:
                    os.close(parent_fd)
                except OSError:
                    pass
    try:
        _wait_bounded_io_child(pid, deadline=deadline, operation="input staging")
    finally:
        if not target.is_file():
            try:
                target.unlink()
            except OSError:
                pass


def _atomic_commit_output_to_scope(
    scope_fd: int,
    relative: Path,
    source: Path,
    *,
    max_bytes: int,
    deadline: float,
) -> None:
    if not source.is_file():
        raise WorkerExecutionError("worker completed without required staged output: {}".format(source))
    if max_bytes <= 0 or source.stat().st_size > max_bytes:
        raise ResourceDenied("output exceeds trusted publication byte limit: {}".format(relative))
    if time.monotonic() >= deadline:
        raise WorkerTimeout("output publication exceeded end-to-end deadline")

    committed_read, committed_write = os.pipe()
    os.set_blocking(committed_read, False)
    pid = os.fork()
    if pid == 0:
        os.close(committed_read)
        parent_fd = None
        fd = None
        temp_name = None
        committed = False
        try:
            parent_fd, name = _open_parent_from_scope(scope_fd, relative, create_missing=True)
            temp_name = ".{}.sunokiller-{}".format(name, uuid.uuid4().hex)
            fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
            copied = 0
            with source.open("rb") as staged, os.fdopen(os.dup(fd), "wb") as destination:
                while True:
                    chunk = staged.read(min(1024 * 1024, max_bytes - copied + 1))
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > max_bytes:
                        os._exit(2)
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            os.close(fd)
            fd = None
            os.replace(temp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            committed = True
            os.write(committed_write, b"C")
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
            os._exit(0)
        except BaseException:
            os._exit(1)
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if not committed and parent_fd is not None and temp_name is not None:
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                except OSError:
                    pass
            if parent_fd is not None:
                try:
                    os.close(parent_fd)
                except OSError:
                    pass
            try:
                os.close(committed_write)
            except OSError:
                pass
    os.close(committed_write)
    try:
        _wait_bounded_io_child(pid, deadline=deadline, operation="output publication", committed_fd=committed_read)
    finally:
        os.close(committed_read)



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
        deadline: float,
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
        self._deadline = deadline

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
                _copy_regular_input_from_scope(
                    scope_fd,
                    relative,
                    staged,
                    max_bytes=self.policy.max_input_bytes,
                    deadline=self._deadline,
                )
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
            _atomic_commit_output_to_scope(
                scope_fd,
                relative,
                staged,
                max_bytes=self.policy.max_output_bytes,
                deadline=self._deadline,
            )

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

    The child receives only the JSON job payload plus trusted finite resource
    limits and a minimized environment. It cannot write canonical state
    directly; it may only propose `_state`, which the parent commits
    transactionally after successful execution.

    Security contract:
    - exact worker code target must exist in the trusted worker-policy registry;
    - the signed lease subject must equal that exact module:function worker_id;
    - capability/logical-resource policy comes only from the trusted registry;
    - caller resource limits must be finite, positive, and no greater than the
      trusted per-worker maxima;
    - the child starts with Python isolated mode and a pinned trusted package
      source root, never ambient CWD/PYTHONPATH import resolution;
    - POSIX CPU/memory limits are applied inside worker_entry after exec, not
      through preexec_fn in the multithreaded parent;
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
        trusted_executables: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.authority = authority
        self.state_store = state_store
        self.state_key = state_key
        self.trusted_executables = dict(trusted_executables or {})

    @staticmethod
    def _policy_for(worker: WorkerSpec) -> WorkerPolicy:
        policy = _TRUSTED_WORKER_POLICIES.get(worker.worker_id)
        if policy is None:
            raise WorkerExecutionError(
                "worker is not registered in trusted policy: {}".format(worker.worker_id)
            )
        return policy

    @staticmethod
    def _validated_limits(worker: WorkerSpec, policy: WorkerPolicy) -> Dict[str, Any]:
        timeout = worker.timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
            or float(timeout) > float(policy.max_timeout_seconds)
        ):
            raise WorkerExecutionError(
                "timeout_seconds must be finite, positive, and <= {} for {}".format(
                    policy.max_timeout_seconds, policy.worker_id
                )
            )

        memory = worker.max_memory_mb
        if (
            isinstance(memory, bool)
            or not isinstance(memory, int)
            or memory <= 0
            or memory > policy.max_memory_mb
        ):
            raise WorkerExecutionError(
                "max_memory_mb must be a positive integer <= {} for {}".format(
                    policy.max_memory_mb, policy.worker_id
                )
            )

        cpu = worker.max_cpu_seconds
        if (
            isinstance(cpu, bool)
            or not isinstance(cpu, int)
            or cpu <= 0
            or cpu > policy.max_cpu_seconds
        ):
            raise WorkerExecutionError(
                "max_cpu_seconds must be a positive integer <= {} for {}".format(
                    policy.max_cpu_seconds, policy.worker_id
                )
            )

        return {
            "timeout_seconds": float(timeout),
            "max_memory_mb": int(memory),
            "max_cpu_seconds": int(cpu),
        }

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
        limits: Mapping[str, Any],
    ) -> Tuple[int, str, str]:
        args = list(_isolated_worker_command())
        kwargs: Dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "env": _minimal_environment(),
        }

        if os.name == "posix":
            # start_new_session is implemented by subprocess without running a
            # Python preexec callback. Resource limits are applied after exec in
            # the trusted worker_entry process, avoiding fork-time deadlocks.
            kwargs["start_new_session"] = True
        elif os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        process = subprocess.Popen(args, **kwargs)
        try:
            stdout, stderr = process.communicate(
                json.dumps(dict(envelope)),
                timeout=float(limits["timeout_seconds"]),
            )
        except subprocess.TimeoutExpired as exc:
            _terminate_process_tree(process)
            try:
                stdout, stderr = process.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process)
                try:
                    stdout, stderr = process.communicate(timeout=1.0)
                except subprocess.TimeoutExpired:
                    # Never turn timeout recovery into an unbounded pipe drain.
                    for stream in (process.stdin, process.stdout, process.stderr):
                        if stream is not None:
                            try:
                                stream.close()
                            except OSError:
                                pass
                    stdout, stderr = "", "worker pipes did not close after process-group kill"
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
        limits = self._validated_limits(worker, policy)

        pre = self.state_store.load_latest(self.state_key)
        pre_hash = pre.state_hash if pre else digest_json({})
        started = int(time.time())
        input_hash = digest_json(input_payload)
        deadline = time.monotonic() + float(limits["timeout_seconds"])

        with _FilesystemStager(
            lease=lease,
            policy=policy,
            payload=input_payload,
            deadline=deadline,
        ) as stager:
            if policy.worker_id == "sunokiller.omen:mastering_worker" and not input_payload.get("dry_run"):
                ffmpeg = self.trusted_executables.get("ffmpeg")
                if not ffmpeg:
                    raise WorkerExecutionError("trusted ffmpeg executable is not configured")
                ffmpeg_path = Path(ffmpeg)
                if (
                    not ffmpeg_path.is_absolute()
                    or ffmpeg_path.is_symlink()
                    or not ffmpeg_path.is_file()
                    or not os.access(str(ffmpeg_path), os.X_OK)
                ):
                    raise WorkerExecutionError("trusted ffmpeg executable must be an absolute executable regular file")
                stager.execution_payload["_trusted_ffmpeg_path"] = str(ffmpeg_path)

            envelope = {
                "module": worker.module,
                "function": worker.function,
                "payload": stager.execution_payload,
                "limits": {
                    "max_memory_mb": limits["max_memory_mb"],
                    "max_cpu_seconds": limits["max_cpu_seconds"],
                },
            }

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeout("end-to-end execution deadline expired during staging")
            worker_limits = dict(limits)
            worker_limits["timeout_seconds"] = remaining
            returncode, stdout, stderr = self._run_worker(
                worker=worker,
                envelope=envelope,
                limits=worker_limits,
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
            self._validated_limits(worker, policy)

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
