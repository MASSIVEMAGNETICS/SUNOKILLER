import dataclasses
import os
import tempfile
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
from sunokiller.runtime.state import StateConflict
from sunokiller.omen import build_master_command


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

    def issue(self, ttl=300):
        return self.authority.issue_lease(
            subject="victor-daw-master-worker",
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
            subject="worker",
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

    def test_receipt_tampering_rejected(self):
        runner = IsolatedRunner(
            authority=self.authority,
            state_store=self.store,
            state_key="runtime",
        )
        lease = self.authority.issue_lease(
            subject="demo",
            capabilities=["audio.master"],
            resource_scopes=["catalog/masters"],
        )
        result, receipt = runner.execute(
            lease=lease,
            worker=WorkerSpec(
                module="sunokiller.runtime.demo_worker",
                function="update_counter",
                capability="audio.master",
                resource="catalog/masters/demo",
                timeout_seconds=5,
                max_memory_mb=None,
                max_cpu_seconds=None,
            ),
            payload={"value": 4},
        )
        self.assertEqual(result["value"], 5)
        self.authority.verify_receipt(receipt)
        tampered = dataclasses.replace(receipt, output_hash="0" * 64)
        with self.assertRaises(InvalidSignature):
            self.authority.verify_receipt(tampered)


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


if __name__ == "__main__":
    unittest.main()
