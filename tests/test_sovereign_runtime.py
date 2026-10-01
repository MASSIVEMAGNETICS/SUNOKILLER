"""Security-contract regressions for the bounded sovereign runtime."""

import dataclasses
import fcntl
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import sunokiller.runtime.contracts as contracts_module
import sunokiller.runtime.runner as runner_module
from sunokiller.runtime import HMACAuthority, IsolatedRunner, SQLiteStateStore, WorkerSpec
from sunokiller.runtime.contracts import (
    CapabilityDenied,
    InvalidSignature,
    LeaseExpired,
    LeaseRevoked,
    ResourceDenied,
)
from sunokiller.runtime.runner import (
    WorkerExecutionError,
    WorkerTimeout,
    _atomic_commit_output_to_scope,
    _stage_regular_input_anonymous_bounded,
    _open_directory_no_symlinks,
    _open_directory_bounded,
)
from sunokiller.runtime.state import (
    NO_SNAPSHOT_PRECONDITION,
    StateConflict,
    StateDeadlineExceeded,
)
from sunokiller.omen import OmenError, build_master_command


class CapabilityBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.authority = HMACAuthority(b"x" * 32)
        handle = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.db_path = handle.name
        handle.close()
        self.store = SQLiteStateStore(self.db_path)

    def tearDown(self):
        self.store.close()
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def runner(self, trusted_executables=None):
        return IsolatedRunner(
            authority=self.authority,
            state_store=self.store,
            state_key="runtime",
            trusted_executables=trusted_executables,
        )

    def demo_worker(self):
        return WorkerSpec(
            module="sunokiller.runtime.demo_worker",
            function="update_counter",
            timeout_seconds=5,
            max_memory_mb=512,
            max_cpu_seconds=5,
        )

    def omen_worker(self, timeout_seconds=5):
        # No caller-provided capability/resource/path policy exists here.
        # Those requirements come from the trusted runtime registry.
        return WorkerSpec(
            module="sunokiller.omen",
            function="mastering_worker",
            timeout_seconds=timeout_seconds,
            max_memory_mb=512,
            max_cpu_seconds=5,
        )

    def issue(self, ttl=300):
        return self.authority.issue_lease(
            subject=self.demo_worker().worker_id,
            capabilities=["audio.master"],
            resource_scopes=["catalog/masters"],
            ttl_seconds=ttl,
        )

    def test_authorized_lease(self):
        lease = self.issue()
        self.authority.verify_lease(
            lease,
            required_capability="audio.master",
            resource="catalog/masters/exit_velocity.wav",
        )

    def test_unauthorized_capability_rejected(self):
        lease = self.issue()
        with self.assertRaises(CapabilityDenied):
            self.authority.verify_lease(
                lease,
                required_capability="audio.synthesize",
                resource="catalog/masters/exit_velocity.wav",
            )

    def test_out_of_scope_resource_rejected(self):
        lease = self.issue()
        with self.assertRaises(ResourceDenied):
            self.authority.verify_lease(
                lease,
                required_capability="audio.master",
                resource="private/identity/state.json",
            )

    def test_expired_lease_rejected(self):
        now = int(time.time())
        lease = self.authority.issue_lease(
            subject=self.demo_worker().worker_id,
            capabilities=["audio.master"],
            resource_scopes=["catalog"],
            ttl_seconds=1,
            not_before=now - 10,
        )
        with self.assertRaises(LeaseExpired):
            self.authority.verify_lease(
                lease,
                required_capability="audio.master",
                resource="catalog/x.wav",
                now=now,
            )

    def test_revoked_lease_rejected(self):
        lease = self.issue()
        self.store.revoke_lease(lease.lease_id, "human stop")
        with self.assertRaises(LeaseRevoked):
            self.authority.verify_lease(
                lease,
                required_capability="audio.master",
                resource="catalog/masters/x.wav",
                revoked_ids=self.store.revoked_ids(),
            )

    def test_tampered_lease_rejected(self):
        lease = self.issue()
        tampered = dataclasses.replace(lease, subject="attacker")
        with self.assertRaises(InvalidSignature):
            self.authority.verify_lease(
                tampered,
                required_capability="audio.master",
                resource="catalog/masters/x.wav",
            )

    def test_oversized_forged_lease_is_rejected_before_copy_hash_or_state_read(self):
        lease = dataclasses.replace(
            self.issue(),
            metadata={"attacker": "x" * (65 * 1024)},
        )
        with mock.patch.object(
            contracts_module.json,
            "dumps",
            side_effect=AssertionError("oversized lease must fail before JSON encoding"),
        ), mock.patch.object(
            runner_module.os,
            "fork",
            side_effect=AssertionError("oversized lease must fail before helper fork"),
        ), mock.patch.object(
            self.store,
            "is_revoked",
            side_effect=AssertionError("forged lease must not trigger SQLite I/O"),
        ):
            with self.assertRaises(InvalidSignature):
                self.runner().execute(
                    lease=lease,
                    worker=self.demo_worker(),
                    payload={"value": 4},
                )

    def test_mutable_authority_fields_cannot_reuse_a_valid_signature(self):
        lease = self.issue()
        forged_capabilities = dataclasses.replace(
            lease,
            capabilities=list(lease.capabilities),
        )
        forged_scopes = dataclasses.replace(
            lease,
            resource_scopes=list(lease.resource_scopes),
        )
        for forged in (forged_capabilities, forged_scopes):
            with self.assertRaises(InvalidSignature):
                self.authority.verify_lease(
                    forged,
                    required_capability="audio.master",
                    resource="catalog/masters/x.wav",
                )

    def test_state_mismatch_rejected(self):
        first = self.store.save_snapshot("runtime", {"v": 1})
        self.store.save_snapshot("runtime", {"v": 2}, expected_hash=first.state_hash)
        with self.assertRaises(StateConflict):
            self.store.save_snapshot("runtime", {"v": 3}, expected_hash=first.state_hash)

    def test_empty_state_precondition_rejected_after_first_writer(self):
        first = self.store.save_snapshot(
            "fresh-runtime",
            {"writer": 1},
            expected_hash=NO_SNAPSHOT_PRECONDITION,
        )
        self.assertEqual(first.version, 1)
        with self.assertRaises(StateConflict):
            self.store.save_snapshot(
                "fresh-runtime",
                {"writer": 2},
                expected_hash=NO_SNAPSHOT_PRECONDITION,
            )

    def test_revocation_uses_state_store_lock(self):
        lease = self.issue()
        started = threading.Event()
        finished = threading.Event()

        def revoke():
            started.set()
            self.store.revoke_lease(lease.lease_id, "human stop")
            finished.set()

        with self.store._lock:
            thread = threading.Thread(target=revoke)
            thread.start()
            self.assertTrue(started.wait(timeout=1.0))
            self.assertFalse(finished.wait(timeout=0.05))
        thread.join(timeout=1.0)
        self.assertTrue(finished.is_set())
        self.assertTrue(self.store.is_revoked(lease.lease_id))

    def test_expiry_is_rechecked_after_sqlite_write_lock_is_acquired(self):
        lease = self.issue(ttl=1)
        blocker = sqlite3.connect(self.db_path, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        failures = []

        def write_after_wait():
            try:
                self.store.save_snapshot(
                    "expiry-runtime",
                    {"value": 1},
                    expected_hash=NO_SNAPSHOT_PRECONDITION,
                    lease_id=lease.lease_id,
                    lease_expires_at=lease.expires_at,
                )
            except Exception as exc:
                failures.append(exc)

        thread = threading.Thread(target=write_after_wait)
        thread.start()
        time.sleep(1.2)
        blocker.execute("COMMIT")
        blocker.close()
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(failures)
        self.assertIsInstance(failures[0], LeaseExpired)
        self.assertIsNone(self.store.load_latest("expiry-runtime"))

    def test_deadline_bound_state_commit_is_refused_before_transaction(self):
        started = time.monotonic()
        with mock.patch.object(
            self.store,
            "_immediate_transaction",
            side_effect=AssertionError("SQLite transaction must not begin"),
        ):
            with self.assertRaises(StateDeadlineExceeded):
                self.store.save_snapshot(
                    "runtime",
                    {"value": 5},
                    deadline_monotonic=time.monotonic() + 1.0,
                )
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertIsNone(self.store.load_latest("runtime"))

    def test_worker_proposed_state_is_rejected_without_sqlite_transaction(self):
        lease = self.issue()
        with mock.patch.object(
            self.store,
            "save_snapshot",
            side_effect=AssertionError("state commit must not execute"),
        ):
            with self.assertRaisesRegex(
                WorkerExecutionError,
                "worker-proposed state mutation is disabled",
            ):
                self.runner().execute(
                    lease=lease,
                    worker=self.demo_worker(),
                    payload={"value": 4, "propose_state": True},
                )
        self.assertIsNone(self.store.load_latest("runtime"))

    def test_signed_lease_is_bound_to_exact_worker_code(self):
        lease = self.issue()
        with self.assertRaises(WorkerExecutionError):
            self.runner().execute(
                lease=lease,
                worker=self.omen_worker(),
                payload={"input_path": "x", "output_path": "y"},
            )

    def test_payload_size_limit_blocks_worker_launch(self):
        lease = self.issue()
        started = time.monotonic()
        with mock.patch.object(
            IsolatedRunner,
            "_run_worker",
            side_effect=AssertionError("oversized payload must not launch"),
        ), mock.patch.object(
            runner_module.os,
            "fork",
            side_effect=AssertionError("oversized payload must fail before fork"),
        ):
            with self.assertRaises(ResourceDenied):
                self.runner().execute(
                    lease=lease,
                    worker=self.demo_worker(),
                    payload={"value": 4, "unused": "x" * (65 * 1024)},
                )
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertIsNone(self.store.load_latest("runtime"))

    def test_exhausted_aggregate_admission_cannot_reach_helper_fork(self):
        gate = threading.BoundedSemaphore(1)
        self.assertTrue(gate.acquire(blocking=False))
        worker = dataclasses.replace(self.demo_worker(), timeout_seconds=0.05)
        started = time.monotonic()
        try:
            with mock.patch.object(
                runner_module,
                "_EXECUTION_ADMISSION",
                gate,
            ), mock.patch.object(
                runner_module.os,
                "fork",
                side_effect=AssertionError("capacity exhaustion must precede fork"),
            ):
                with self.assertRaises(WorkerTimeout):
                    self.runner().execute(
                        lease=self.issue(),
                        worker=worker,
                        payload={"value": 4},
                    )
        finally:
            gate.release()
        self.assertLess(time.monotonic() - started, 1.0)

    def test_aggregate_admission_is_retained_until_unwind_then_released(self):
        gate = threading.BoundedSemaphore(1)

        def assert_slot_held(*args, **kwargs):
            self.assertFalse(gate.acquire(blocking=False))
            raise WorkerExecutionError("injected admitted-path failure")

        with mock.patch.object(
            runner_module,
            "_EXECUTION_ADMISSION",
            gate,
        ), mock.patch.object(
            runner_module,
            "_canonicalize_payload_bounded",
            side_effect=assert_slot_held,
        ):
            with self.assertRaisesRegex(WorkerExecutionError, "injected admitted-path"):
                self.runner().execute(
                    lease=self.issue(),
                    worker=self.demo_worker(),
                    payload={"value": 4},
                )
        self.assertTrue(gate.acquire(blocking=False))
        gate.release()

    @unittest.skipUnless(os.name == "posix", "killable state reads require POSIX")
    def test_bounded_reads_remain_bound_to_original_database_after_cwd_change(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            origin = root / "origin"
            decoy = root / "decoy"
            origin.mkdir()
            decoy.mkdir()
            old_cwd = os.getcwd()
            bound_store = None
            decoy_store = None
            try:
                os.chdir(origin)
                bound_store = SQLiteStateStore("runtime.sqlite3")
                lease = self.issue()
                bound_store.revoke_lease(lease.lease_id, "human stop")

                os.chdir(decoy)
                decoy_store = SQLiteStateStore("runtime.sqlite3")
                decoy_store.close()
                decoy_store = None
                runner = IsolatedRunner(
                    authority=self.authority,
                    state_store=bound_store,
                    state_key="runtime",
                )
                with mock.patch.object(
                    IsolatedRunner,
                    "_run_worker",
                    side_effect=AssertionError("revoked worker must not launch"),
                ):
                    with self.assertRaises(LeaseRevoked):
                        runner.execute(
                            lease=lease,
                            worker=self.demo_worker(),
                            payload={"value": 4},
                        )
            finally:
                os.chdir(old_cwd)
                if decoy_store is not None:
                    decoy_store.close()
                if bound_store is not None:
                    bound_store.close()

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "object-bound state reads require Linux proc descriptors",
    )
    def test_bounded_reads_follow_open_database_object_across_path_swap(self):
        with tempfile.TemporaryDirectory() as work:
            database = Path(work) / "runtime.sqlite3"
            original_object = Path(work) / "original.sqlite3"
            bound_store = SQLiteStateStore(str(database))
            decoy_store = None
            try:
                lease = self.issue()
                bound_store.revoke_lease(lease.lease_id, "human stop")
                bound_store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                os.replace(database, original_object)

                decoy_store = SQLiteStateStore(str(database))
                decoy_store.close()
                decoy_store = None

                self.assertTrue(
                    bound_store.is_revoked(
                        lease.lease_id,
                        deadline_monotonic=time.monotonic() + 1.0,
                    )
                )
            finally:
                if decoy_store is not None:
                    decoy_store.close()
                bound_store.close()

    @unittest.skipUnless(os.name == "posix", "killable state reads require POSIX")
    def test_in_memory_state_store_fails_closed_for_bounded_execution(self):
        memory_store = SQLiteStateStore(":memory:")
        try:
            runner = IsolatedRunner(
                authority=self.authority,
                state_store=memory_store,
                state_key="runtime",
            )
            with mock.patch.object(
                IsolatedRunner,
                "_run_worker",
                side_effect=AssertionError("worker must not launch without bound state identity"),
            ):
                with self.assertRaises(WorkerTimeout):
                    runner.execute(
                        lease=self.issue(),
                        worker=self.demo_worker(),
                        payload={"value": 4},
                    )
        finally:
            memory_store.close()

    @unittest.skipUnless(os.name == "posix", "killable state reads require POSIX")
    def test_blocking_sqlite_read_cannot_delay_human_stop_deadline(self):
        lease = self.issue()
        worker = dataclasses.replace(self.demo_worker(), timeout_seconds=0.1)
        real_connect = sqlite3.connect

        def blocking_connect(*args, **kwargs):
            time.sleep(5)
            return real_connect(*args, **kwargs)

        started = time.monotonic()
        with mock.patch.object(sqlite3, "connect", blocking_connect), mock.patch.object(
            IsolatedRunner,
            "_run_worker",
            side_effect=AssertionError("worker must not launch after state timeout"),
        ):
            with self.assertRaises(WorkerTimeout):
                self.runner().execute(
                    lease=lease,
                    worker=worker,
                    payload={"value": 4},
                )
        self.assertLess(time.monotonic() - started, 1.0)

    def test_unregistered_worker_is_rejected_before_execution(self):
        worker = WorkerSpec(
            module="os.path",
            function="exists",
        )
        lease = self.authority.issue_lease(
            subject=worker.worker_id,
            capabilities=["audio.master"],
            resource_scopes=["*"],
        )
        with self.assertRaises(WorkerExecutionError):
            self.runner().execute(lease=lease, worker=worker, payload={})

    def test_caller_cannot_disable_or_raise_trusted_execution_limits(self):
        lease = self.issue()
        base = self.demo_worker()
        invalid_workers = [
            dataclasses.replace(base, timeout_seconds=None),
            dataclasses.replace(base, timeout_seconds=31),
            dataclasses.replace(base, max_memory_mb=None),
            dataclasses.replace(base, max_memory_mb=1025),
            dataclasses.replace(base, max_cpu_seconds=None),
            dataclasses.replace(base, max_cpu_seconds=31),
        ]
        for worker in invalid_workers:
            with self.subTest(worker=worker):
                with self.assertRaises(WorkerExecutionError):
                    self.runner().execute(lease=lease, worker=worker, payload={"value": 4})
        self.assertIsNone(self.store.load_latest("runtime"))

    def test_ambient_pythonpath_and_cwd_cannot_shadow_worker_entry(self):
        runner = self.runner()
        lease = self.issue()
        with tempfile.TemporaryDirectory() as shadow:
            root = Path(shadow)
            package = root / "sunokiller" / "runtime"
            package.mkdir(parents=True)
            (root / "sunokiller" / "__init__.py").write_text("", encoding="utf-8")
            (package / "__init__.py").write_text("", encoding="utf-8")
            marker = root / "shadow-worker-entry-ran.txt"
            (package / "worker_entry.py").write_text(
                "from pathlib import Path\nPath({!r}).write_text('owned')\n".format(str(marker)),
                encoding="utf-8",
            )

            old_cwd = os.getcwd()
            old_pythonpath = os.environ.get("PYTHONPATH")
            os.chdir(root)
            os.environ["PYTHONPATH"] = str(root)
            try:
                result, receipt = runner.execute(
                    lease=lease,
                    worker=self.demo_worker(),
                    payload={"value": 4},
                )
            finally:
                os.chdir(old_cwd)
                if old_pythonpath is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = old_pythonpath

            self.assertEqual(result["value"], 5)
            self.assertFalse(marker.exists())
            self.authority.verify_receipt(receipt)

    def test_human_stop_during_worker_blocks_canonical_commit(self):
        runner = self.runner()
        lease = self.issue()
        failures = []

        def execute_slow_worker():
            try:
                runner.execute(
                    lease=lease,
                    worker=self.demo_worker(),
                    payload={"value": 4, "sleep_seconds": 0.5},
                )
            except Exception as exc:
                failures.append(exc)

        thread = threading.Thread(target=execute_slow_worker)
        thread.start()
        time.sleep(0.15)
        self.store.revoke_lease(lease.lease_id, "human stop during execution")
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(failures)
        self.assertIsInstance(failures[0], LeaseRevoked)
        self.assertIsNone(self.store.load_latest("runtime"))

    def test_receipt_tampering_rejected(self):
        runner = self.runner()
        lease = self.issue()
        result, receipt = runner.execute(
            lease=lease,
            worker=self.demo_worker(),
            payload={"value": 4},
        )
        self.assertEqual(result["value"], 5)
        self.authority.verify_receipt(receipt)
        tampered = dataclasses.replace(receipt, output_hash="0" * 64)
        with self.assertRaises(InvalidSignature):
            self.authority.verify_receipt(tampered)

    def test_execution_ids_unique_for_identical_invocations(self):
        runner = self.runner()
        lease = self.issue()
        _, first = runner.execute(
            lease=lease,
            worker=self.demo_worker(),
            payload={"value": 4},
        )
        _, second = runner.execute(
            lease=lease,
            worker=self.demo_worker(),
            payload={"value": 4},
        )
        self.assertNotEqual(first.execution_id, second.execution_id)
        self.authority.verify_receipt(first)
        self.authority.verify_receipt(second)

    @unittest.skipUnless(os.name == "posix", "descriptor-safe filesystem broker is POSIX v0.1")
    def test_omen_paths_are_enforced_by_trusted_policy_without_caller_fields(self):
        runner = self.runner()
        omen_worker = self.omen_worker()
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as forbidden:
            source = Path(allowed) / "in.wav"
            source.write_bytes(b"placeholder")
            forbidden_output = Path(forbidden) / "out.wav"
            lease = self.authority.issue_lease(
                subject=omen_worker.worker_id,
                capabilities=["audio.master"],
                resource_scopes=["catalog/masters", str(Path(allowed).resolve())],
            )
            with self.assertRaises(ResourceDenied):
                runner.execute(
                    lease=lease,
                    worker=omen_worker,
                    payload={
                        "input_path": str(source),
                        "output_path": str(forbidden_output),
                        "dry_run": True,
                    },
                )

    @unittest.skipUnless(os.name == "posix", "descriptor-safe filesystem broker is POSIX v0.1")
    def test_filesystem_colon_sibling_does_not_escape_scope(self):
        runner = self.runner()
        omen_worker = self.omen_worker()
        with tempfile.TemporaryDirectory() as parent:
            parent_path = Path(parent)
            allowed = parent_path / "allowed"
            sibling = parent_path / "allowed:outside"
            allowed.mkdir()
            sibling.mkdir()
            source = allowed / "in.wav"
            source.write_bytes(b"placeholder")
            forbidden_output = sibling / "out.wav"
            lease = self.authority.issue_lease(
                subject=omen_worker.worker_id,
                capabilities=["audio.master"],
                resource_scopes=["catalog/masters", str(allowed.resolve())],
            )
            with self.assertRaises(ResourceDenied):
                runner.execute(
                    lease=lease,
                    worker=omen_worker,
                    payload={
                        "input_path": str(source),
                        "output_path": str(forbidden_output),
                        "dry_run": True,
                    },
                )

    @unittest.skipUnless(os.name == "posix", "descriptor-safe filesystem broker is POSIX v0.1")
    def test_symlink_component_is_rejected_before_worker_execution(self):
        runner = self.runner()
        omen_worker = self.omen_worker()
        with tempfile.TemporaryDirectory() as allowed:
            root = Path(allowed).resolve()
            real_dir = root / "real"
            real_dir.mkdir()
            source = real_dir / "in.wav"
            source.write_bytes(b"placeholder")
            link = root / "route"
            link.symlink_to(real_dir, target_is_directory=True)
            output = root / "out.wav"
            lease = self.authority.issue_lease(
                subject=omen_worker.worker_id,
                capabilities=["audio.master"],
                resource_scopes=["catalog/masters", str(root)],
            )
            with self.assertRaises(ResourceDenied):
                runner.execute(
                    lease=lease,
                    worker=omen_worker,
                    payload={
                        "input_path": str(link / "in.wav"),
                        "output_path": str(output),
                        "dry_run": True,
                    },
                )

    @unittest.skipUnless(os.name == "posix", "process-group regression uses POSIX sessions")
    def test_worker_timeout_kills_descendant_process_group(self):
        with tempfile.TemporaryDirectory() as work:
            marker = Path(work) / "descendant-survived.txt"
            script = Path(work) / "worker.py"
            script.write_text(
                "import json, subprocess, sys, time\n"
                "json.load(sys.stdin)\n"
                "subprocess.Popen([sys.executable, '-c', "
                + repr(
                    "import time; from pathlib import Path; "
                    "time.sleep(0.8); Path({!r}).write_text('survived')".format(
                        str(marker)
                    )
                )
                + "])\n"
                "time.sleep(5)\n",
                encoding="utf-8",
            )
            runner = self.runner()
            with mock.patch.object(
                runner_module,
                "_isolated_worker_command",
                return_value=(sys.executable, str(script)),
            ):
                with self.assertRaises(WorkerTimeout):
                    runner._run_worker(
                        worker=dataclasses.replace(
                            self.demo_worker(), timeout_seconds=0.15
                        ),
                        envelope={},
                        limits={"timeout_seconds": 0.15},
                    )
            time.sleep(1.0)
            self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "posix", "process-group regression uses POSIX sessions")
    def test_timeout_kills_group_when_worker_leader_already_exited(self):
        with tempfile.TemporaryDirectory() as work:
            marker = Path(work) / "orphan-survived.txt"
            script = Path(work) / "worker.py"
            script.write_text(
                "import json, subprocess, sys\n"
                "json.load(sys.stdin)\n"
                "subprocess.Popen([sys.executable, '-c', "
                + repr(
                    "import time; from pathlib import Path; "
                    "time.sleep(0.8); Path({!r}).write_text('survived')".format(
                        str(marker)
                    )
                )
                + "])\n",
                encoding="utf-8",
            )
            runner = self.runner()
            with mock.patch.object(
                runner_module,
                "_isolated_worker_command",
                return_value=(sys.executable, str(script)),
            ):
                with self.assertRaises(WorkerTimeout):
                    runner._run_worker(
                        worker=dataclasses.replace(
                            self.demo_worker(), timeout_seconds=0.15
                        ),
                        envelope={},
                        limits={"timeout_seconds": 0.15},
                    )
            time.sleep(1.0)
            self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "posix", "native execution fail-closed test uses POSIX")
    def test_bounded_runtime_never_executes_configured_or_ambient_ffmpeg(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as fake_bin:
            root = Path(allowed).resolve()
            source = root / "in.wav"
            source.write_bytes(b"audio")
            output = root / "out.wav"
            marker = root / "ffmpeg-ran.txt"
            fake_ffmpeg = Path(fake_bin) / "ffmpeg"
            fake_ffmpeg.write_text(
                "#!{}\nfrom pathlib import Path\nPath({!r}).write_text('ran')\n".format(
                    sys.executable, str(marker)
                ),
                encoding="utf-8",
            )
            fake_ffmpeg.chmod(0o755)
            runner = self.runner({"ffmpeg": str(fake_ffmpeg)})
            lease = self.authority.issue_lease(
                subject=self.omen_worker().worker_id,
                capabilities=["audio.master"],
                resource_scopes=["catalog/masters", str(root)],
            )
            with mock.patch.object(
                runner_module.subprocess,
                "Popen",
                side_effect=AssertionError("no worker or FFmpeg may launch"),
            ):
                with self.assertRaisesRegex(
                    WorkerExecutionError,
                    "native file-producing execution is disabled",
                ):
                    runner.execute(
                        lease=lease,
                        worker=self.omen_worker(),
                        payload={"input_path": str(source), "output_path": str(output)},
                    )
            self.assertFalse(output.exists())
            self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "posix", "descriptor-safe filesystem broker is POSIX v0.1")
    def test_input_staging_enforces_trusted_byte_limit(self):
        policy_id = self.omen_worker().worker_id
        limited = dataclasses.replace(
            runner_module._TRUSTED_WORKER_POLICIES[policy_id],
            max_input_bytes=4,
        )
        with tempfile.TemporaryDirectory() as allowed, mock.patch.dict(
            runner_module._TRUSTED_WORKER_POLICIES,
            {policy_id: limited},
        ):
            root = Path(allowed).resolve()
            source = root / "in.wav"
            source.write_bytes(b"12345")
            output = root / "out.wav"
            lease = self.authority.issue_lease(
                subject=policy_id,
                capabilities=["audio.master"],
                resource_scopes=["catalog/masters", str(root)],
            )
            with self.assertRaises(ResourceDenied):
                self.runner().execute(
                    lease=lease,
                    worker=self.omen_worker(),
                    payload={
                        "input_path": str(source),
                        "output_path": str(output),
                        "dry_run": True,
                    },
                )
            self.assertFalse(output.exists())

    @unittest.skipUnless(
        sys.platform.startswith("linux") and hasattr(os, "memfd_create"),
        "anonymous staging requires Linux memfd",
    )
    def test_blocking_anonymous_staging_creation_is_bounded(self):
        runner = self.runner()
        omen_worker = self.omen_worker(timeout_seconds=0.1)
        with tempfile.TemporaryDirectory() as allowed:
            root = Path(allowed).resolve()
            source = root / "in.wav"
            source.write_bytes(b"audio")
            output = root / "out.wav"
            lease = self.authority.issue_lease(
                subject=omen_worker.worker_id,
                capabilities=["audio.master"],
                resource_scopes=["catalog/masters", str(root)],
            )
            real_memfd_create = os.memfd_create

            def blocking_memfd_create(*args, **kwargs):
                time.sleep(5)
                return real_memfd_create(*args, **kwargs)

            started = time.monotonic()
            with mock.patch.object(
                os, "memfd_create", blocking_memfd_create
            ), mock.patch.object(
                runner_module.subprocess,
                "Popen",
                side_effect=AssertionError("worker must not launch"),
            ):
                with self.assertRaises(WorkerTimeout):
                    runner.execute(
                        lease=lease,
                        worker=omen_worker,
                        payload={
                            "input_path": str(source),
                            "output_path": str(output),
                            "dry_run": True,
                        },
                    )
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertFalse(output.exists())

    @unittest.skipUnless(
        sys.platform.startswith("linux") and hasattr(os, "memfd_create"),
        "sealed anonymous input requires Linux memfd",
    )
    def test_anonymous_dry_run_token_is_empty_sealed_and_never_reads_input(self):
        with tempfile.TemporaryDirectory() as allowed:
            root = Path(allowed).resolve()
            (root / "in.wav").write_bytes(b"trusted-input")
            scope_fd = _open_directory_no_symlinks(root)
            staged_fd = None
            try:
                with mock.patch.object(
                    tempfile,
                    "mkdtemp",
                    side_effect=AssertionError("pathname staging must not execute"),
                ), mock.patch.object(
                    os,
                    "read",
                    side_effect=AssertionError("dry-run staging must not copy input bytes"),
                ):
                    staged_fd = _stage_regular_input_anonymous_bounded(
                        scope_fd,
                        Path("in.wav"),
                        max_bytes=1024,
                        deadline=time.monotonic() + 1.0,
                    )
                required = runner_module._required_input_seals()
                self.assertEqual(
                    fcntl.fcntl(
                        staged_fd,
                        runner_module._fcntl_seal_command("F_GET_SEALS"),
                    )
                    & required,
                    required,
                )
                self.assertEqual(os.pread(staged_fd, 64, 0), b"")
                with self.assertRaises(OSError):
                    os.pwrite(staged_fd, b"attacker", 0)
            finally:
                if staged_fd is not None:
                    os.close(staged_fd)
                os.close(scope_fd)

    @unittest.skipUnless(os.name == "posix", "disabled output path contract is POSIX v0.1")
    def test_disabled_output_symlink_is_lexical_only_and_never_published(self):
        runner = self.runner()
        omen_worker = self.omen_worker()
        with tempfile.TemporaryDirectory() as allowed:
            root = Path(allowed).resolve()
            source = root / "in.wav"
            source.write_bytes(b"placeholder")
            destination = root / "destination"
            destination.mkdir()
            output_link = root / "output-link"
            output_link.symlink_to(destination, target_is_directory=True)
            output = output_link / "out.wav"
            lease = self.authority.issue_lease(
                subject=omen_worker.worker_id,
                capabilities=["audio.master"],
                resource_scopes=["catalog/masters", str(root)],
            )
            result, receipt = runner.execute(
                lease=lease,
                worker=omen_worker,
                payload={
                    "input_path": str(source),
                    "output_path": str(output),
                    "dry_run": True,
                },
            )
            self.assertEqual(result["status"], "DRY_RUN")
            self.assertEqual(result["output"], str(output))
            self.authority.verify_receipt(receipt)
            self.assertFalse((destination / "out.wav").exists())

    @unittest.skipUnless(
        sys.platform.startswith("linux") and hasattr(os, "memfd_create"),
        "sealed anonymous input requires Linux memfd",
    )
    def test_dry_run_staging_does_not_copy_input_bytes(self):
        with tempfile.TemporaryDirectory() as allowed:
            root = Path(allowed).resolve()
            (root / "in.wav").write_bytes(b"audio")
            scope_fd = _open_directory_no_symlinks(root)
            started = time.monotonic()
            staged_fd = None
            try:
                with mock.patch.object(
                    os,
                    "read",
                    side_effect=AssertionError("input bytes must remain outside the worker"),
                ):
                    staged_fd = _stage_regular_input_anonymous_bounded(
                        scope_fd,
                        Path("in.wav"),
                        max_bytes=1024,
                        deadline=time.monotonic() + 0.5,
                    )
                self.assertEqual(os.fstat(staged_fd).st_size, 0)
            finally:
                if staged_fd is not None:
                    os.close(staged_fd)
                os.close(scope_fd)
            self.assertLess(time.monotonic() - started, 1.0)

    @unittest.skipUnless(os.name == "posix", "bounded scope transfer uses POSIX descriptors")
    def test_blocking_signed_scope_traversal_is_bounded(self):
        real_open = os.open

        def blocking_root_open(path, *args, **kwargs):
            if path == "/":
                time.sleep(5)
            return real_open(path, *args, **kwargs)

        started = time.monotonic()
        with mock.patch.object(os, "open", blocking_root_open):
            with self.assertRaises(WorkerTimeout):
                _open_directory_bounded(
                    Path("/authorized"),
                    deadline=time.monotonic() + 0.1,
                )
        self.assertLess(time.monotonic() - started, 1.0)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and hasattr(os, "memfd_create"),
        "sealed anonymous input requires Linux memfd",
    )
    def test_blocking_input_descriptor_preflight_is_bounded(self):
        with tempfile.TemporaryDirectory() as allowed:
            root = Path(allowed).resolve()
            (root / "in.wav").write_bytes(b"audio")
            scope_fd = _open_directory_no_symlinks(root)
            real_open = os.open

            def blocking_open(path, *args, **kwargs):
                if path == "in.wav" and kwargs.get("dir_fd") is not None:
                    time.sleep(5)
                return real_open(path, *args, **kwargs)

            started = time.monotonic()
            try:
                with mock.patch.object(os, "open", blocking_open):
                    with self.assertRaises(WorkerTimeout):
                        _stage_regular_input_anonymous_bounded(
                            scope_fd,
                            Path("in.wav"),
                            max_bytes=1024,
                            deadline=time.monotonic() + 0.1,
                        )
            finally:
                os.close(scope_fd)
            self.assertLess(time.monotonic() - started, 1.0)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and hasattr(os, "memfd_create"),
        "sealed anonymous input requires Linux memfd",
    )
    def test_input_timeout_has_no_parent_filesystem_cleanup_path(self):
        with tempfile.TemporaryDirectory() as allowed:
            root = Path(allowed).resolve()
            (root / "in.wav").write_bytes(b"audio")
            scope_fd = _open_directory_no_symlinks(root)
            real_fstat = os.fstat
            parent_pid = os.getpid()

            def blocking_fstat(*args, **kwargs):
                time.sleep(5)
                return real_fstat(*args, **kwargs)

            def forbidden_parent_path_call(*args, **kwargs):
                if os.getpid() == parent_pid:
                    raise AssertionError("parent filesystem cleanup path executed")
                return False

            started = time.monotonic()
            try:
                with mock.patch.object(os, "fstat", blocking_fstat), mock.patch.object(
                    Path, "is_file", forbidden_parent_path_call
                ), mock.patch.object(Path, "unlink", forbidden_parent_path_call):
                    with self.assertRaises(WorkerTimeout):
                        _stage_regular_input_anonymous_bounded(
                            scope_fd,
                            Path("in.wav"),
                            max_bytes=1024,
                            deadline=time.monotonic() + 0.1,
                        )
            finally:
                os.close(scope_fd)
            self.assertLess(time.monotonic() - started, 1.0)

    @unittest.skipUnless(os.name == "posix", "public publication boundary is POSIX v0.1")
    def test_publication_fails_closed_without_visible_output_or_receipt(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as staging:
            root = Path(allowed).resolve()
            source = Path(staging) / "master.wav"
            source.write_bytes(b"master")
            output = root / "out.wav"
            scope_fd = _open_directory_no_symlinks(root)
            try:
                with self.assertRaisesRegex(
                    WorkerExecutionError, "durable receipt-linked transaction"
                ):
                    _atomic_commit_output_to_scope(
                        scope_fd,
                        Path("out.wav"),
                        source,
                        max_bytes=1024,
                        deadline=time.monotonic() + 1.0,
                    )
            finally:
                os.close(scope_fd)
            self.assertFalse(output.exists())

    @unittest.skipUnless(os.name == "posix", "public publication boundary is POSIX v0.1")
    def test_forced_preemption_window_has_no_public_replace_operation(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as staging:
            root = Path(allowed).resolve()
            source = Path(staging) / "master.wav"
            source.write_bytes(b"master")
            output = root / "out.wav"
            scope_fd = _open_directory_no_symlinks(root)
            try:
                with mock.patch.object(
                    os, "replace", side_effect=AssertionError("replace must not execute")
                ):
                    with self.assertRaises(WorkerExecutionError):
                        _atomic_commit_output_to_scope(
                            scope_fd,
                            Path("out.wav"),
                            source,
                            max_bytes=1024,
                            deadline=time.monotonic() + 1.0,
                        )
            finally:
                os.close(scope_fd)
            self.assertFalse(output.exists())

    @unittest.skipUnless(os.name == "posix", "descriptor-safe filesystem broker is POSIX v0.1")
    def test_omen_dry_run_succeeds_when_actual_paths_are_in_scope(self):
        runner = self.runner()
        omen_worker = self.omen_worker()
        with tempfile.TemporaryDirectory() as allowed:
            root = Path(allowed).resolve()
            source = root / "in.wav"
            output = root / "out.wav"
            source.write_bytes(b"placeholder")
            lease = self.authority.issue_lease(
                subject=omen_worker.worker_id,
                capabilities=["audio.master"],
                resource_scopes=["catalog/masters", str(root)],
            )
            result, receipt = runner.execute(
                lease=lease,
                worker=omen_worker,
                payload={
                    "input_path": str(source),
                    "output_path": str(output),
                    "dry_run": True,
                },
            )
            self.assertEqual(result["status"], "DRY_RUN")
            self.assertEqual(result["sample_rate"], 48000)
            self.assertEqual(result["input"], str(source))
            self.assertEqual(result["output"], str(output))
            self.authority.verify_receipt(receipt)

    @unittest.skipUnless(os.name == "posix", "descriptor-safe filesystem broker is POSIX v0.1")
    def test_nested_descriptor_paths_are_restored_and_receipt_hash_is_stable(self):
        runner = self.runner()
        omen_worker = self.omen_worker()
        with tempfile.TemporaryDirectory() as allowed:
            root = Path(allowed).resolve()
            source = root / "in.wav"
            output = root / "out.wav"
            source.write_bytes(b"placeholder")
            lease = self.authority.issue_lease(
                subject=omen_worker.worker_id,
                capabilities=["audio.master"],
                resource_scopes=["catalog/masters", str(root)],
            )
            payload = {
                "input_path": str(source),
                "output_path": str(output),
                "dry_run": True,
            }
            first_result, first_receipt = runner.execute(
                lease=lease,
                worker=omen_worker,
                payload=payload,
            )
            padding_fds = [os.open("/dev/null", os.O_RDONLY) for _ in range(8)]
            try:
                second_result, second_receipt = runner.execute(
                    lease=lease,
                    worker=omen_worker,
                    payload=payload,
                )
            finally:
                for fd in padding_fds:
                    os.close(fd)

            self.assertEqual(first_result, second_result)
            self.assertEqual(first_receipt.output_hash, second_receipt.output_hash)
            self.assertIn(str(source), first_result["command"])
            self.assertIn(str(output), first_result["command"])
            self.assertFalse(
                any("/proc/self/fd/" in item for item in first_result["command"])
            )
            self.authority.verify_receipt(first_receipt)
            self.authority.verify_receipt(second_receipt)

    def test_descriptor_restoration_includes_nested_mapping_keys(self):
        stager = object.__new__(runner_module._FilesystemStager)
        stager._public_path_map = {"/proc/self/fd/91": "/trusted/input.wav"}
        result = stager.restore_public_paths(
            {"nested": {"/proc/self/fd/91": ["/proc/self/fd/91"]}}
        )
        self.assertEqual(
            result,
            {"nested": {"/trusted/input.wav": ["/trusted/input.wav"]}},
        )


class OmenHarnessTests(unittest.TestCase):
    def test_command_enforces_48k_and_loudnorm(self):
        cmd = build_master_command(
            input_path="in.wav",
            output_path="out.wav",
            target_lufs=-14.0,
            sample_rate=48000,
        )
        rendered = " ".join(cmd)
        self.assertIn("-ar 48000", rendered)
        self.assertIn("loudnorm=I=-14.0", rendered)
        self.assertIn("pcm_s24le", rendered)

    def test_non_48k_override_is_rejected(self):
        with self.assertRaises(OmenError):
            build_master_command(
                input_path="in.wav",
                output_path="out.wav",
                sample_rate=44100,
            )


if __name__ == "__main__":
    unittest.main()
