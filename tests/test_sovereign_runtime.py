import dataclasses
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
import unittest

from sunokiller.runtime import HMACAuthority, IsolatedRunner, SQLiteStateStore, WorkerSpec
from sunokiller.runtime.contracts import (
    CapabilityDenied,
    InvalidSignature,
    LeaseExpired,
    LeaseRevoked,
    ResourceDenied,
)
from sunokiller.runtime.runner import WorkerExecutionError, WorkerTimeout
from sunokiller.runtime.state import NO_SNAPSHOT_PRECONDITION, StateConflict
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

    def runner(self):
        return IsolatedRunner(
            authority=self.authority,
            state_store=self.store,
            state_key="runtime",
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

    def test_signed_lease_is_bound_to_exact_worker_code(self):
        lease = self.issue()
        with self.assertRaises(WorkerExecutionError):
            self.runner().execute(
                lease=lease,
                worker=self.omen_worker(),
                payload={"input_path": "x", "output_path": "y"},
            )

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

    @unittest.skipUnless(os.name == "posix", "process-group regression uses POSIX executable semantics")
    def test_worker_timeout_kills_ffmpeg_descendant(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as fake_bin:
            root = Path(allowed).resolve()
            source = root / "in.wav"
            source.write_bytes(b"placeholder")
            output = root / "out.wav"
            marker = root / "descendant-survived.txt"

            fake_ffmpeg = Path(fake_bin) / "ffmpeg"
            fake_ffmpeg.write_text(
                "#!{}\n"
                "import time\n"
                "from pathlib import Path\n"
                "time.sleep(0.8)\n"
                "Path({!r}).write_text('survived')\n".format(sys.executable, str(marker)),
                encoding="utf-8",
            )
            fake_ffmpeg.chmod(0o755)

            omen_worker = self.omen_worker(timeout_seconds=0.15)
            lease = self.authority.issue_lease(
                subject=omen_worker.worker_id,
                capabilities=["audio.master"],
                resource_scopes=["catalog/masters", str(root)],
            )

            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = str(fake_bin) + os.pathsep + old_path
            try:
                with self.assertRaises(WorkerTimeout):
                    runner.execute(
                        lease=lease,
                        worker=omen_worker,
                        payload={
                            "input_path": str(source),
                            "output_path": str(output),
                        },
                    )
            finally:
                os.environ["PATH"] = old_path

            time.sleep(1.0)
            self.assertFalse(marker.exists())
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
