# Sovereign Runtime Boundary v0.1

This branch implements the first clean, bounded execution slice for the Victor DAW / SUNOKILLER stack. It is a successor architecture, not a repair of legacy Victor monoliths.

## Invariants implemented

1. **Identity/state external to workers** — canonical runtime state is stored in SQLite WAL snapshots, not model weights or worker memory.
2. **Signed capability leases** — a worker must present a signed lease binding issuer, subject, capabilities, resource scope, validity window, nonce, and metadata.
3. **Revocation / Human STOP path** — lease IDs can be revoked in the external state store and are checked before execution.
4. **Process isolation** — work runs in a separate Python interpreter with a minimized environment and optional POSIX memory/CPU limits.
5. **Transactional state commit** — workers may only propose `_state`; the parent process writes it with an optimistic pre-state hash check.
6. **Signed execution receipts** — successful executions produce a signed receipt containing worker identity, capability/resource, input/output hashes, and pre/post state hashes.
7. **Fail closed** — expired, revoked, unsigned/tampered, wrong-capability, and out-of-scope leases are rejected before worker execution.

## Acceptance matrix

| Case | Expected result |
|---|---|
| Valid signed lease + allowed capability/resource | Worker runs |
| Wrong capability | Reject |
| Out-of-scope resource | Reject |
| Expired lease | Reject |
| Revoked lease / Human STOP | Reject |
| Tampered lease | Reject |
| State precondition mismatch | Reject state commit |
| Tampered receipt | Reject receipt verification |
| OMEN master command | 48 kHz + EBU R128 `loudnorm` contract |

`tests/test_sovereign_runtime.py` exercises these paths.

## OMEN mastering worker

`omen` is exposed as a CLI and as `sunokiller.omen:mastering_worker` for capability-leased execution.

Example:

```bash
omen mix.wav -o master.wav --target-lufs -14 --true-peak -1
```

The first harness uses FFmpeg's EBU R128 `loudnorm` filter, forces 48 kHz, and emits WAV 24-bit PCM, FLAC, or 320 kbps MP3 based on extension.

## Deliberate boundaries / not yet claimed

- **HMAC-SHA256 is the v0.1 signing mechanism.** It provides a real signed contract with a secret held outside repository state, but it is symmetric. The contract is designed so Ed25519 can replace it later.
- **Process isolation is not a full container/seccomp sandbox.** POSIX CPU/memory limits are applied where supported. Stronger OS sandboxing is a follow-on hardening step.
- **Stem separation is not reimplemented here.** Existing GAWDCORE/stem-separation source must be verified and then registered as a separate leased worker. OMEN can master a mix or a pre-rendered stem mix now.
- **The Gemini continuity harness is complementary evidence, not the security gate.** Model-swap state-delta semantics can be integrated after this capability boundary passes independent review.

## Next integration slice

1. Register the current perception worker (`ai_ear` / sensory river) behind `audio.perceive`.
2. Register synthesis behind `audio.synthesize`.
3. Verify the existing stem separator and register it behind `audio.separate_stems`.
4. Run OMEN behind `audio.master`.
5. Chain receipts so each stage proves the exact input hash consumed from the prior stage.
