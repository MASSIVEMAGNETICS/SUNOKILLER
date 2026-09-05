# Sovereign Runtime Boundary v0.1

This branch implements the first clean, bounded execution slice for the Victor DAW / SUNOKILLER stack. It is a successor architecture, not a repair of legacy Victor monoliths.

## Invariants implemented

1. **Identity/state external to workers** — canonical runtime state is stored in SQLite WAL snapshots, not model weights or worker memory.
2. **Signed capability leases** — a worker must present a signed lease binding issuer, exact worker subject, capabilities, resource scope, validity window, nonce, and metadata.
3. **Trusted worker policy registry** — callers select only a registered `module:function` target and bounded execution limits. Capability, logical resource, and required filesystem fields come from the trusted runtime registry, not caller-controlled labels.
4. **Revocation / Human STOP path** — lease revocation uses the same locked SQLite transaction boundary as state persistence; a revoked lease cannot commit canonical state after a long-running worker returns.
5. **Process isolation** — work runs in a separate Python interpreter with a minimized environment and optional POSIX memory/CPU limits.
6. **Transactional state commit** — workers may only propose `_state`; the parent process writes it with optimistic pre-state hashing, including an explicit no-snapshot first-writer precondition.
7. **Typed resource enforcement** — logical namespace scopes and filesystem scopes use separate matching rules. Filesystem access requires absolute resolved paths and real descendant containment; logical `:` namespace syntax is never used for path authorization.
8. **Signed execution receipts** — successful executions produce a signed receipt containing the trusted worker identity, capability/resource, input/output hashes, and pre/post state hashes.
9. **Fail closed** — expired, revoked, unsigned/tampered, wrong-capability, out-of-scope, unregistered-worker, worker-substitution, and filesystem-escape attempts are rejected.

## Acceptance matrix

| Case | Expected result |
|---|---|
| Valid signed lease + registered worker + allowed resources | Worker runs |
| Wrong capability | Reject |
| Out-of-scope logical resource | Reject |
| Out-of-scope filesystem path | Reject |
| Colon-sibling filesystem escape | Reject |
| Unregistered worker | Reject |
| Signed lease subject / worker mismatch | Reject |
| Expired lease | Reject |
| Revoked lease / Human STOP | Reject |
| Human STOP during child execution | Reject result; no canonical state commit |
| Tampered lease | Reject |
| Concurrent first-writer / state precondition mismatch | Reject stale state commit |
| Tampered receipt | Reject receipt verification |
| Duplicate same-input executions | Unique receipt IDs |
| OMEN master command | Hard 48 kHz + EBU R128 `loudnorm` contract |
| OMEN 44.1 kHz override | Reject |

`tests/test_sovereign_runtime.py` exercises these paths. A reconstructed current-head execution pass completed 19/19 security/contract checks before final independent review.

## OMEN mastering worker

`omen` is exposed as a CLI and as the trusted worker `sunokiller.omen:mastering_worker` for capability-leased execution. The runtime policy requires both `input_path` and `output_path` to fall under signed absolute filesystem scopes.

Example:

```bash
omen mix.wav -o master.wav --target-lufs -14 --true-peak -1
```

The first harness uses FFmpeg's EBU R128 `loudnorm` filter, forces 48 kHz, and emits WAV 24-bit PCM, FLAC, or 320 kbps MP3 based on extension.

## Deliberate boundaries / not yet claimed

- **HMAC-SHA256 is the v0.1 signing mechanism.** It provides a real signed contract with a secret held outside repository state, but it is symmetric. The contract is designed so Ed25519 can replace it later.
- **The trusted worker registry is static in v0.1.** A signed manifest registry with code/package digests is the intended hardening path for dynamic workers.
- **Process isolation is not a full container/seccomp sandbox.** POSIX CPU/memory limits are applied where supported. Stronger OS sandboxing is a follow-on hardening step.
- **Stem separation is not reimplemented here.** Existing GAWDCORE/stem-separation source must be verified and then registered as a separate leased worker. OMEN can master a mix or a pre-rendered stem mix now.
- **The Gemini continuity harness is complementary evidence, not the security gate.** Model-swap state-delta semantics can be integrated after this capability boundary passes independent review.

## Next integration slice

1. Register the current perception worker (`ai_ear` / sensory river) behind `audio.perceive`.
2. Register synthesis behind `audio.synthesize`.
3. Verify the existing stem separator and register it behind `audio.separate_stems`.
4. Run OMEN behind `audio.master`.
5. Chain receipts so each stage proves the exact input hash consumed from the prior stage.
