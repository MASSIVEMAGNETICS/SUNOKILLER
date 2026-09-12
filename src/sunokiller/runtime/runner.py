"""Process-isolated execution boundary for capability-leased work."""

from __future__ import annotations

from dataclasses import dataclass
import array
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
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
from .state import NO_SNAPSHOT_PRECONDITION, SQLiteStateStore, StateDeadlineExceeded


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
    max_payload_bytes: int = 64 * 1024
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


def _canonicalize_payload_bounded(
    payload: Mapping[str, Any],
    *,
    max_bytes: int,
    deadline: float,
) -> Tuple[Dict[str, Any], str]:
    """Canonicalize untrusted payload data in a killable, size-bounded child."""
    if os.name != "posix" or not hasattr(os, "fork"):
        raise WorkerExecutionError(
            "bounded payload canonicalization requires POSIX process isolation in v0.1"
        )
    if max_bytes <= 0:
        raise WorkerExecutionError("trusted payload byte limit must be positive")

    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    pid = os.fork()
    if pid == 0:
        parent_sock.close()
        try:
            _preflight_json_size(payload, max_bytes=max_bytes)
            encoder = json.JSONEncoder(
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            total = 0
            for text_chunk in encoder.iterencode(payload):
                encoded = text_chunk.encode("utf-8")
                total += len(encoded)
                if total > max_bytes:
                    os._exit(2)
                child_sock.sendall(encoded)
            os._exit(0)
        except ResourceDenied:
            os._exit(2)
        except BaseException:
            os._exit(1)

    child_sock.close()
    parent_sock.setblocking(False)
    chunks = []
    total = 0
    try:
        while True:
            try:
                chunk = parent_sock.recv(64 * 1024)
                if chunk:
                    total += len(chunk)
                    if total > max_bytes:
                        _bounded_kill_and_reap(pid)
                        raise ResourceDenied("payload exceeded its trusted byte limit")
                    chunks.append(chunk)
            except BlockingIOError:
                pass

            completed, status = os.waitpid(pid, os.WNOHANG)
            if completed == pid:
                while True:
                    try:
                        chunk = parent_sock.recv(64 * 1024)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ResourceDenied("payload exceeded its trusted byte limit")
                    chunks.append(chunk)
                if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 2:
                    raise ResourceDenied("payload exceeded its trusted byte limit")
                if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
                    raise WorkerExecutionError(
                        "payload must be a JSON object within the trusted byte limit"
                    )
                break

            if time.monotonic() >= deadline:
                _bounded_kill_and_reap(pid)
                raise WorkerTimeout(
                    "payload canonicalization exceeded end-to-end deadline"
                )
            time.sleep(0.01)
    finally:
        parent_sock.close()

    raw = b"".join(chunks)
    try:
        normalized = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerExecutionError("payload canonicalization returned invalid JSON") from exc
    if not isinstance(normalized, dict):
        raise WorkerExecutionError("payload must be a JSON object")
    return normalized, sha256(raw).hexdigest()


def _preflight_json_size(value: Any, *, max_bytes: int) -> int:
    """Count canonical JSON bytes without allocating an encoded string token."""
    active = set()

    def add(total: int, amount: int) -> int:
        total += amount
        if total > max_bytes:
            raise ResourceDenied("payload exceeded its trusted byte limit")
        return total

    def string_size(text: str, total: int) -> int:
        total = add(total, 2)
        if len(text) + total > max_bytes:
            raise ResourceDenied("payload exceeded its trusted byte limit")
        for character in text:
            codepoint = ord(character)
            if character in ('"', "\\") or character in "\b\f\n\r\t":
                width = 2
            elif codepoint < 0x20:
                width = 6
            elif codepoint < 0x80:
                width = 1
            elif codepoint < 0x800:
                width = 2
            elif 0xD800 <= codepoint <= 0xDFFF:
                raise ValueError("payload contains an unpaired surrogate")
            elif codepoint < 0x10000:
                width = 3
            else:
                width = 4
            total = add(total, width)
        return total

    def measure(item: Any, total: int, depth: int) -> int:
        if depth > 64:
            raise ValueError("payload nesting exceeds the trusted depth limit")
        if item is None:
            return add(total, 4)
        if item is True:
            return add(total, 4)
        if item is False:
            return add(total, 5)
        if type(item) is str:
            return string_size(item, total)
        if type(item) is int:
            # A binary integer larger than the entire byte budget cannot have a
            # valid decimal JSON representation within that budget.
            if item.bit_length() > max_bytes * 4:
                raise ResourceDenied("payload exceeded its trusted byte limit")
            return add(total, len(str(item)))
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("payload contains a non-finite number")
            return add(total, len(repr(item)))
        if type(item) is dict:
            identity = id(item)
            if identity in active:
                raise ValueError("payload contains a circular reference")
            active.add(identity)
            try:
                total = add(total, 2)
                for index, (key, nested) in enumerate(item.items()):
                    if type(key) is not str:
                        raise ValueError("payload object keys must be strings")
                    if index:
                        total = add(total, 1)
                    total = string_size(key, total)
                    total = add(total, 1)
                    total = measure(nested, total, depth + 1)
                return total
            finally:
                active.remove(identity)
        if type(item) in (list, tuple):
            identity = id(item)
            if identity in active:
                raise ValueError("payload contains a circular reference")
            active.add(identity)
            try:
                total = add(total, 2)
                for index, nested in enumerate(item):
                    if index:
                        total = add(total, 1)
                    total = measure(nested, total, depth + 1)
                return total
            finally:
                active.remove(identity)
        raise ValueError("payload contains a non-JSON value")

    return measure(value, 0, 0)


def _open_directory_bounded(path: Path, *, deadline: float) -> int:
    """Open a signed scope in a killable child and transfer only the verified fd."""
    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    pid = os.fork()
    if pid == 0:
        parent_sock.close()
        fd = None
        try:
            fd = _open_directory_no_symlinks(path)
            rights = array.array("i", [fd])
            child_sock.sendmsg([b"F"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)])
            os._exit(0)
        except BaseException:
            os._exit(1)
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            child_sock.close()

    child_sock.close()
    parent_sock.setblocking(False)
    try:
        while True:
            try:
                data, ancdata, _, _ = parent_sock.recvmsg(1, socket.CMSG_SPACE(array.array("i").itemsize))
                if data == b"F":
                    for level, kind, payload in ancdata:
                        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                            rights = array.array("i")
                            rights.frombytes(payload[: rights.itemsize])
                            received_fd = rights[0]
                            _bounded_kill_and_reap(pid)
                            return received_fd
            except BlockingIOError:
                pass
            completed, status = os.waitpid(pid, os.WNOHANG)
            if completed == pid:
                raise ResourceDenied("descriptor-safe signed scope traversal failed")
            if time.monotonic() >= deadline:
                _bounded_kill_and_reap(pid)
                raise WorkerTimeout("signed scope traversal exceeded end-to-end deadline")
            time.sleep(0.01)
    finally:
        parent_sock.close()


def _require_linux_anonymous_staging() -> None:
    if (
        not sys.platform.startswith("linux")
        or not hasattr(os, "memfd_create")
        or not hasattr(os, "MFD_ALLOW_SEALING")
    ):
        raise WorkerExecutionError(
            "secure anonymous filesystem staging requires Linux memfd sealing in v0.1"
        )


def _required_input_seals() -> int:
    import fcntl

    return (
        getattr(fcntl, "F_SEAL_SEAL", 0x0001)
        | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
        | getattr(fcntl, "F_SEAL_GROW", 0x0004)
        | getattr(fcntl, "F_SEAL_WRITE", 0x0008)
    )


def _fcntl_seal_command(name: str) -> int:
    import fcntl

    # Linux UAPI values; some minimal Python builds omit the symbolic names.
    fallback = {"F_ADD_SEALS": 1033, "F_GET_SEALS": 1034}
    return int(getattr(fcntl, name, fallback[name]))


def _anonymous_descriptor_path(fd: int) -> str:
    _require_linux_anonymous_staging()
    return "/proc/self/fd/{}".format(fd)


def _receive_anonymous_fd_bounded(
    parent_sock: socket.socket,
    pid: int,
    *,
    deadline: float,
    operation: str,
    require_sealed: bool,
    max_bytes: int,
) -> int:
    import fcntl

    parent_sock.setblocking(False)
    control_size = socket.CMSG_SPACE(array.array("i").itemsize)
    try:
        while True:
            try:
                data, ancdata, _, _ = parent_sock.recvmsg(1, control_size)
                if data == b"F":
                    received = []
                    for level, kind, payload in ancdata:
                        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                            rights = array.array("i")
                            usable = len(payload) - (len(payload) % rights.itemsize)
                            rights.frombytes(payload[:usable])
                            received.extend(rights.tolist())
                    if len(received) == 1:
                        received_fd = received[0]
                        _bounded_kill_and_reap(pid)
                        info = os.fstat(received_fd)
                        if (
                            not stat.S_ISREG(info.st_mode)
                            or info.st_size > max_bytes
                            or info.st_nlink != 0
                        ):
                            os.close(received_fd)
                            raise ResourceDenied(
                                "{} failed anonymous-file checks".format(operation)
                            )
                        if require_sealed:
                            seals = fcntl.fcntl(
                                received_fd,
                                _fcntl_seal_command("F_GET_SEALS"),
                            )
                            required = _required_input_seals()
                            if seals & required != required:
                                os.close(received_fd)
                                raise ResourceDenied(
                                    "{} was not sealed before transfer".format(operation)
                                )
                        return received_fd
                    for received_fd in received:
                        os.close(received_fd)
                    raise WorkerExecutionError(
                        "{} descriptor transfer was invalid".format(operation)
                    )
            except BlockingIOError:
                pass

            completed, status = os.waitpid(pid, os.WNOHANG)
            if completed == pid:
                if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 2:
                    raise ResourceDenied(
                        "{} exceeded its trusted byte limit".format(operation)
                    )
                if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 3:
                    raise ResourceDenied(
                        "{} failed descriptor containment checks".format(operation)
                    )
                raise WorkerExecutionError(
                    "{} failed in bounded anonymous staging child".format(operation)
                )
            if time.monotonic() >= deadline:
                _bounded_kill_and_reap(pid)
                raise WorkerTimeout(
                    "{} exceeded end-to-end deadline".format(operation)
                )
            time.sleep(0.01)
    finally:
        parent_sock.close()


def _make_anonymous_staging_file_bounded(
    *,
    label: str,
    max_bytes: int,
    deadline: float,
) -> int:
    """Create an unlinked memory-backed staging object in a bounded child."""
    _require_linux_anonymous_staging()
    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    pid = os.fork()
    if pid == 0:
        parent_sock.close()
        fd = None
        try:
            flags = os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
            fd = os.memfd_create(label, flags=flags)
            os.fchmod(fd, 0o600)
            rights = array.array("i", [fd])
            child_sock.sendmsg([b"F"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)])
            os._exit(0)
        except BaseException:
            os._exit(1)

    child_sock.close()
    return _receive_anonymous_fd_bounded(
        parent_sock,
        pid,
        deadline=deadline,
        operation="anonymous output staging creation",
        require_sealed=False,
        max_bytes=max_bytes,
    )


def _stage_regular_input_anonymous_bounded(
    scope_fd: int,
    relative: Path,
    *,
    max_bytes: int,
    deadline: float,
) -> int:
    """Validate input and transfer an empty sealed dry-run capability token.

    Bounded v0.1 never launches FFmpeg and the OMEN dry-run worker only checks
    that its descriptor is a regular file. Copying audio bytes into memfd would
    consume uncharged shmem without adding verification value, so no source byte
    crosses this boundary.
    """
    _require_linux_anonymous_staging()
    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    pid = os.fork()
    if pid == 0:
        parent_sock.close()
        parent_fd = None
        source_fd = None
        staged_fd = None
        try:
            import fcntl

            parent_fd, name = _open_parent_from_scope(
                scope_fd,
                relative,
                create_missing=False,
            )
            source_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            info = os.fstat(source_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                os._exit(3)
            if max_bytes <= 0 or info.st_size > max_bytes:
                os._exit(2)

            staged_fd = os.memfd_create(
                "sunokiller-input",
                flags=os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
            )
            os.fchmod(staged_fd, 0o400)
            fcntl.fcntl(
                staged_fd,
                _fcntl_seal_command("F_ADD_SEALS"),
                _required_input_seals(),
            )
            rights = array.array("i", [staged_fd])
            child_sock.sendmsg([b"F"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)])
            os._exit(0)
        except (OSError, ResourceDenied):
            os._exit(3)
        except BaseException:
            os._exit(1)

    child_sock.close()
    return _receive_anonymous_fd_bounded(
        parent_sock,
        pid,
        deadline=deadline,
        operation="sealed anonymous input staging",
        require_sealed=True,
        max_bytes=max_bytes,
    )


def _atomic_commit_output_to_scope(
    scope_fd: int,
    relative: Path,
    source: Path,
    *,
    max_bytes: int,
    deadline: float,
) -> None:
    """Fail closed until visibility and receipt commit share one durable transaction.

    POSIX rename and parent acknowledgment cannot be made atomic across process
    preemption. Publishing here could therefore expose an output without a
    matching success receipt. The bounded v0.1 runtime intentionally refuses
    public output publication instead of overstating transactional safety.
    """
    del scope_fd, relative, source, max_bytes, deadline
    raise WorkerExecutionError(
        "public output publication is disabled until a durable receipt-linked "
        "transaction is implemented"
    )



class _FilesystemStager:
    """Broker filesystem inputs through sealed anonymous descriptors.

    The worker never receives a caller-controlled authorized path. A killable
    child validates input metadata and gives the dry-run worker only an empty,
    sealed Linux memfd token; protected input bytes never enter the worker.
    Output receives an anonymous memfd only for dry-run contract construction;
    bounded v0.1 rejects every native file-producing execution before launch.
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
        self._scope_fds = []
        self._staging_fds = []
        self._public_path_map = {}
        self._deadline = deadline

    def __enter__(self) -> "_FilesystemStager":
        if not self.policy.filesystem_fields:
            return self

        _require_linux_anonymous_staging()
        try:
            for field in self.policy.filesystem_inputs:
                raw = self.original_payload.get(field)
                if not isinstance(raw, str) or not raw:
                    raise WorkerExecutionError(
                        "payload resource field {} must be a non-empty path string".format(field)
                    )
                scope, relative = _select_signed_filesystem_scope(self.lease, raw)
                scope_fd = _open_directory_bounded(scope, deadline=self._deadline)
                self._scope_fds.append(scope_fd)
                staged_fd = _stage_regular_input_anonymous_bounded(
                    scope_fd,
                    relative,
                    max_bytes=self.policy.max_input_bytes,
                    deadline=self._deadline,
                )
                self._staging_fds.append(staged_fd)
                staged_path = _anonymous_descriptor_path(staged_fd)
                self.execution_payload[field] = staged_path
                self._public_path_map[staged_path] = str(_normalized_absolute_path(raw))

            for field in self.policy.filesystem_outputs:
                raw = self.original_payload.get(field)
                if not isinstance(raw, str) or not raw:
                    raise WorkerExecutionError(
                        "payload resource field {} must be a non-empty path string".format(field)
                    )
                staged_fd = _make_anonymous_staging_file_bounded(
                    label="sunokiller-output",
                    max_bytes=self.policy.max_output_bytes,
                    deadline=self._deadline,
                )
                self._staging_fds.append(staged_fd)
                staged_path = _anonymous_descriptor_path(staged_fd)
                self.execution_payload[field] = staged_path
                self.execution_payload["_staged_output_suffix"] = Path(raw).suffix
                self._public_path_map[staged_path] = str(_normalized_absolute_path(raw))

            self.execution_payload["_descriptor_staging_fds"] = list(self._staging_fds)
        except OSError as exc:
            self._cleanup()
            raise ResourceDenied("anonymous descriptor staging failed") from exc
        except Exception:
            self._cleanup()
            raise

        return self

    def commit_outputs(self) -> None:
        if not self.policy.filesystem_outputs or self.original_payload.get("dry_run") is True:
            return
        raise WorkerExecutionError(
            "public output publication is disabled until a durable receipt-linked "
            "transaction is implemented"
        )

    @property
    def pass_fds(self) -> Tuple[int, ...]:
        return tuple(self._staging_fds)

    def restore_public_paths(self, result: Dict[str, Any]) -> Dict[str, Any]:
        def restore(value: Any) -> Any:
            if isinstance(value, str):
                return self._public_path_map.get(value, value)
            if isinstance(value, list):
                return [restore(item) for item in value]
            if isinstance(value, dict):
                return {key: restore(item) for key, item in value.items()}
            return value

        return restore(dict(result))

    def _cleanup(self) -> None:
        for fd in self._scope_fds + self._staging_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._scope_fds.clear()
        self._staging_fds.clear()

    def __exit__(self, exc_type, exc, tb) -> None:
        self._cleanup()


class IsolatedRunner:
    """Run a registered, capability-leased worker in a separate interpreter.

    The child receives only a canonical byte-bounded JSON payload plus trusted
    finite resource limits and a minimized environment. Bounded v0.1 rejects
    worker-proposed state mutation and native file-producing execution because
    neither SQLite finalization nor external publication is yet receipt-atomic
    under forced preemption.

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
    - payload canonicalization is process-isolated, deadline-bound, and byte-limited;
    - filesystem dry-run inputs use empty sealed anonymous Linux descriptors;
    - the worker never receives the original authorized filesystem paths;
    - lease validity is rechecked after the subprocess returns and before receipt;
    - worker-proposed state and native output execution fail closed without a write transaction;
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
        policy: Optional[WorkerPolicy] = None,
        deadline: Optional[float] = None,
    ) -> WorkerPolicy:
        policy = policy or self._policy_for(worker)
        if lease.subject != policy.worker_id:
            raise WorkerExecutionError(
                "signed lease subject {} does not authorize worker {}".format(
                    lease.subject, policy.worker_id
                )
            )

        try:
            revoked = self.state_store.is_revoked(
                lease.lease_id,
                deadline_monotonic=deadline,
            )
        except StateDeadlineExceeded as exc:
            raise WorkerTimeout(
                "end-to-end execution deadline expired during authority check"
            ) from exc
        self.authority.verify_lease(
            lease,
            required_capability=policy.capability,
            resource=policy.logical_resource,
            revoked_ids={lease.lease_id} if revoked else (),
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
        pass_fds: Tuple[int, ...] = (),
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
            if pass_fds:
                kwargs["pass_fds"] = pass_fds
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
        policy = self._policy_for(worker)
        limits = self._validated_limits(worker, policy)
        deadline = time.monotonic() + float(limits["timeout_seconds"])
        started = int(time.time())
        input_payload, input_hash = _canonicalize_payload_bounded(
            payload,
            max_bytes=policy.max_payload_bytes,
            deadline=deadline,
        )

        policy = self._verify_worker_and_resources(
            lease=lease,
            worker=worker,
            payload=input_payload,
            policy=policy,
            deadline=deadline,
        )
        try:
            pre_hash = self.state_store.load_latest_hash(
                self.state_key,
                deadline_monotonic=deadline,
            )
        except StateDeadlineExceeded as exc:
            raise WorkerTimeout(
                "end-to-end execution deadline expired while loading state"
            ) from exc
        pre_hash = pre_hash or digest_json({})

        if policy.filesystem_outputs and input_payload.get("dry_run") is not True:
            raise WorkerExecutionError(
                "native file-producing execution is disabled in bounded v0.1; "
                "only dry-run contract evaluation is authorized"
            )

        with _FilesystemStager(
            lease=lease,
            policy=policy,
            payload=input_payload,
            deadline=deadline,
        ) as stager:
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
                pass_fds=stager.pass_fds,
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

            policy = self._verify_worker_and_resources(
                lease=lease,
                worker=worker,
                payload=input_payload,
                policy=policy,
                deadline=deadline,
            )
            self._validated_limits(worker, policy)

            result = dict(message["result"])
            proposed_state = result.pop("_state", None)
            if proposed_state is not None:
                raise WorkerExecutionError(
                    "worker-proposed state mutation is disabled until durable "
                    "transaction finalization can be deadline-supervised"
                )

            stager.commit_outputs()
            result = stager.restore_public_paths(result)

            try:
                self.state_store.assert_lease_active(
                    lease.lease_id,
                    expires_at=lease.expires_at,
                    deadline_monotonic=deadline,
                )
            except StateDeadlineExceeded as exc:
                raise WorkerTimeout(
                    "end-to-end execution deadline expired during final authority check"
                ) from exc

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
            post_state_hash=pre_hash,
            input_hash=input_hash,
            output_hash=digest_json(result),
            status="SUCCESS",
        )
        return result, self.authority.sign_receipt(receipt)
