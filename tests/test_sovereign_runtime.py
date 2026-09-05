import dataclasses
import os
from pathlib import Path
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
from sunokiller.runtime.runner import WorkerExecutionError
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
            max_memory_mb=None,
            max_cpu_seconds=None,
        )

    def omen_worker(self):
        # No caller-provided capability/resource/path policy exists here.
        # Those requirements come from the trusted runtime registry.
        return WorkerSpec(
            module="sunokiller.omen",
            function="mastering_worker",
            timeout_seconds=5,
            max_memory_mb=None,
            max_cpu_seconds=None,
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
            max_memory_mb=None,
            max_cpu_seconds=None,
        )
        lease = self.authority.issue_lease(
            subject=worker.worker_id,
            capabilities=["audio.master"],
            resource_scopes=["*"],
        )
        with self.assertRaises(WorkerExecutionError):
            self.runner().execute(lease=lease, worker=worker, payload={})

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
