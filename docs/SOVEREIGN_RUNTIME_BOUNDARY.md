# Sovereign Runtime Boundary v0.1

This branch implements the first clean, bounded execution slice for the Victor DAW / SUNOKILLER stack. It is a successor architecture, not a repair of legacy Victor monoliths.

## Invariants implemented

1. **Identity/state external to workers** — canonical runtime state is stored in SQLite WAL snapshots, not model weights or worker memory.
2. **Signed capability leases** — a worker must present a signed lease binding issuer, exact worker subject, capabilities, resource scope, validity window, nonce, and metadata.
3. **Trusted worker policy registry** — callers select only a registered `module:function` target and may request only stricter finite execution budgets. Capability, logical resource, required filesystem inputs/outputs, and maximum wall-clock/memory/CPU budgets come from the trusted runtime registry, not caller-controlled labels.
4. **Revocation / Human STOP path** — lease revocation shares the same SQLite serialization boundary as state commits and short external-output finalization. A STOP that lands first blocks the side effect; a side effect that already acquired the lease commit guard finishes before the revocation can commit.
5. **Isolated, pinned child bootstrap** — the child uses Python isolated mode (`-I`), does not inherit `PYTHONPATH`, excludes ambient CWD/user-site import resolution, and inserts only the source root derived from the already-loaded trusted runner before executing `sunokiller.runtime.worker_entry`. A lease cannot redirect the registered worker through an attacker-writable shadow package.
6. **Bounded process execution without fork callbacks** — finite per-worker wall-clock, memory, and CPU maxima are trusted policy. POSIX CPU/address-space limits are applied inside the already-exec'd `worker_entry` before importing the registered worker, not through Python `preexec_fn`; the worker runs in its own process group/session and timeout terminates the whole descendant group.
7. **Transactional state commit** — workers may only propose `_state`; the parent writes it with optimistic pre-state hashing, including an explicit no-snapshot first-writer precondition. Revocation and lease expiry are rechecked after `BEGIN IMMEDIATE` acquires the SQLite write lock.
8. **Typed resource enforcement** — logical namespace scopes and filesystem scopes are distinct. The OMEN filesystem boundary is descriptor-based on POSIX: signed scope directories are traversed with `O_DIRECTORY|O_NOFOLLOW`, inputs are copied from no-follow file descriptors into a private staging directory, and outputs are atomically published through retained authorized directory file descriptors. The worker/FFmpeg never receives the caller's authorized filesystem path directly.
9. **Signed execution receipts** — successful executions produce a signed receipt containing the trusted worker identity, capability/resource, input/output hashes, and pre/post state hashes.
10. **Fail closed** — expired, revoked, unsigned/tampered, wrong-capability, out-of-scope, unregistered-worker, worker-substitution, ambient-import-shadow, disabled/over-max budget, symlink-component, colon-sibling, timeout-descendant, and stale-state attempts are rejected or prevented at the relevant boundary.
11. **Reproducible exact-head verification** — `.github/workflows/sovereign-runtime.yml` compiles and executes the runtime security-contract suite on Ubuntu with Python 3.11 and 3.12 for relevant PR/push changes.

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
| Attacker-writable CWD/PYTHONPATH contains shadow `sunokiller.runtime.worker_entry` | Ignore shadow; execute pinned trusted package entry |
| Caller sets timeout/memory/CPU limit to `None`, non-finite/invalid, or above trusted policy max | Reject before launch |
| Concurrent/multithreaded host launch | No Python `preexec_fn`; POSIX limits apply after exec in trusted child |
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

`tests/test_sovereign_runtime.py` currently defines 24 unit/security-contract tests. The suite contains explicit regressions for the six material findings in the two most recent independent review rounds: descendant process survival after timeout, symlink-swap/path-object containment, lease expiry while waiting for the SQLite write lock, ambient package-entry shadowing, caller-disableable/over-max execution limits, and fork-time `preexec_fn` risk elimination through post-exec child limiting. Passing GitHub Actions on the exact merge head is the reproducible executable gate; earlier reconstructed test passes remain historical evidence only.

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
- **Process isolation is not a full container/seccomp sandbox.** Trusted wall-clock ceilings are always enforced by the parent; POSIX CPU/address-space limits and process-group teardown are enforced where supported. Stronger OS sandboxing is a follow-on hardening step.
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
5. Chain signed receipts so each stage proves the exact input hash consumed from the prior stage.
