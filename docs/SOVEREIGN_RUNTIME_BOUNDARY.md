# Sovereign Runtime Boundary v0.1

This branch implements the first clean, bounded execution slice for the Victor DAW / SUNOKILLER stack. It is a successor architecture, not a repair of legacy Victor monoliths.

## Invariants implemented

1. **Identity/state external to workers** — canonical runtime state is stored in SQLite WAL snapshots, not model weights or worker memory.
2. **Signed capability leases** — a worker must present a signed lease binding issuer, exact worker subject, capabilities, resource scope, validity window, nonce, and metadata.
3. **Trusted worker policy registry** — callers select only a registered `module:function` target and bounded execution limits. Capability, logical resource, and required filesystem inputs/outputs come from the trusted runtime registry, not caller-controlled labels.
4. **Revocation / Human STOP path** — lease revocation shares the same SQLite serialization boundary as state commits and short external-output finalization. A STOP that lands first blocks the side effect; a side effect that already acquired the lease commit guard finishes before the revocation can commit.
5. **Process isolation and bounded teardown** — work runs in a separate Python interpreter with a minimized environment and optional POSIX memory/CPU limits. The worker is launched in its own process group/session; timeout terminates the whole descendant group so an FFmpeg grandchild cannot continue after `WorkerTimeout`.
6. **Transactional state commit** — workers may only propose `_state`; the parent writes it with optimistic pre-state hashing, including an explicit no-snapshot first-writer precondition. Revocation and lease expiry are rechecked after `BEGIN IMMEDIATE` acquires the SQLite write lock.
7. **Typed resource enforcement** — logical namespace scopes and filesystem scopes are distinct. The OMEN filesystem boundary is descriptor-based on POSIX: signed scope directories are traversed with `O_DIRECTORY|O_NOFOLLOW`, inputs are copied from no-follow file descriptors into a private staging directory, and outputs are atomically published through retained authorized directory file descriptors. The worker/FFmpeg never receives the caller's authorized filesystem path directly.
8. **Signed execution receipts** — successful executions produce a signed receipt containing the trusted worker identity, capability/resource, input/output hashes, and pre/post state hashes.
9. **Fail closed** — expired, revoked, unsigned/tampered, wrong-capability, out-of-scope, unregistered-worker, worker-substitution, symlink-component, colon-sibling, timeout-descendant, and stale-state attempts are rejected or prevented at the relevant boundary.
10. **Reproducible exact-head verification** — `.github/workflows/sovereign-runtime.yml` compiles and executes the runtime security-contract suite on Ubuntu with Python 3.11 and 3.12 for relevant PR/push changes.

## Acceptance matrix

| Case | Expected result |
|---|---|
| Valid signed lease + registered worker + allowed resources | Worker runs |
| Wrong capability | Reject |
| Out-of-scope logical resource | Reject |
| Out-of-scope filesystem path | Reject |
| Colon-sibling filesystem escape | Reject |
| Symlink component inside signed filesystem scope | Reject before worker execution |
| Unregistered worker | Reject |
| Signed lease subject / worker mismatch | Reject |
| Expired lease | Reject |
| Lease expires while blocked on SQLite write lock | Reject; no canonical state mutation |
| Revoked lease / Human STOP | Reject |
| Human STOP during child execution | Reject result; no canonical state commit |
| Human STOP racing staged output publication | One serialized outcome: STOP blocks publication or publication completes before STOP commits |
| Worker timeout after spawning FFmpeg | Terminate worker + descendant process group; no surviving output writer |
| Tampered lease | Reject |
| Concurrent first-writer / state precondition mismatch | Reject stale state commit |
| Tampered receipt | Reject receipt verification |
| Duplicate same-input executions | Unique receipt IDs |
| OMEN master command | Hard 48 kHz + EBU R128 `loudnorm` contract |
| OMEN 44.1 kHz override | Reject |

`tests/test_sovereign_runtime.py` currently defines 22 unit/security-contract tests, including explicit regressions for the three material findings reported against the prior exact head: descendant process survival after timeout, symlink-swap/path-object containment, and lease expiry while waiting for the SQLite write lock. Passing GitHub Actions on the exact merge head is the reproducible executable gate; earlier reconstructed test passes remain historical evidence only.

## OMEN mastering worker

`omen` is exposed as a CLI and as the trusted worker `sunokiller.omen:mastering_worker` for capability-leased execution. Under the sovereign runtime, both `input_path` and `output_path` must map lexically beneath a signed absolute filesystem scope and then pass descriptor-based no-follow traversal.

The broker copies the authorized regular input file into a private staging directory. OMEN/FFmpeg consumes that staging path and writes a staging output. After the worker returns and its lease is revalidated, the parent atomically publishes the staged output through a retained authorized directory file descriptor under the short SQLite lease commit guard.

Example standalone CLI use remains:

```bash
omen mix.wav -o master.wav --target-lufs -14 --true-peak -1
```

The first harness uses FFmpeg's EBU R128 `loudnorm` filter, forces 48 kHz, and emits WAV 24-bit PCM, FLAC, or 320 kbps MP3 based on extension.

## Deliberate boundaries / not yet claimed

- **HMAC-SHA256 is the v0.1 signing mechanism.** It provides a real signed contract with a secret held outside repository state, but it is symmetric. The contract is designed so Ed25519 can replace it later.
- **The trusted worker registry is static in v0.1.** A signed manifest registry with code/package digests is the intended hardening path for dynamic workers.
- **Process isolation is not a full container/seccomp sandbox.** POSIX CPU/memory limits and process-group teardown are enforced where supported. Stronger OS sandboxing is a follow-on hardening step.
- **Secure filesystem-worker staging is POSIX-only in v0.1.** Filesystem workers fail closed on platforms without the required descriptor/no-follow primitives. This is deliberate rather than silently falling back to path-string authorization. A Windows implementation should use equivalent OS-enforced handle/reparse-point containment before being enabled.
- **The staging broker currently supports regular-file inputs/outputs.** Inputs with multiple hard links are conservatively rejected to avoid alias ambiguity at this boundary.
- **A worker that combines durable `_state` mutation and external file publication would require a higher-level multi-resource transaction design.** The currently registered OMEN worker is stateless, and the demo state worker has no filesystem outputs; this PR does not claim general atomicity across arbitrary database and filesystem side effects.
- **Stem separation is not reimplemented here.** Existing GAWDCORE/stem-separation source must be verified and then registered as a separate leased worker. OMEN can master a mix or a pre-rendered stem mix now.
- **The Gemini continuity harness is complementary evidence, not the security gate.** Model-swap state-delta semantics can be integrated after this capability boundary passes independent review.

## Verification gate

Before merge/promotion of this bounded runtime slice:

1. The current PR head must be GitHub-mergeable with a clean merge state.
2. The exact-head GitHub Actions `Sovereign Runtime Boundary` matrix must pass on Python 3.11 and 3.12, or the absence/inability of Actions must be recorded explicitly and replaced by an equivalent reproducible execution receipt.
3. Independent Codex review must target the exact merge head; material findings remain blockers until fixed and re-reviewed.
4. The PR description and this document must describe current behavior and limitations, not an earlier tested head.

## Next integration slice

1. Register the current perception worker (`ai_ear` / sensory river) behind `audio.perceive`.
2. Register synthesis behind `audio.synthesize`.
3. Verify the existing stem separator and register it behind `audio.separate_stems`.
4. Run OMEN behind `audio.master`.
5. Chain receipts so each stage proves the exact input hash consumed from the prior stage.
