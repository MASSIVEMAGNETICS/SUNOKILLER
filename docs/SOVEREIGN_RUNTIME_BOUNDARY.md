# Sovereign Runtime Boundary v0.1

This branch implements a deliberately narrow, local-first execution candidate for Victor / SUNOKILLER. It is a successor boundary, not proof of a complete DAW, deployment, mastering delivery, or autonomous authority.

## Current status

- Implementation: built but partial.
- Public output publication: disabled.
- Native FFmpeg execution inside the bounded runtime: disabled.
- Worker-proposed canonical state mutation: disabled.
- Accepted file-producing mode: OMEN contract dry run only.
- Secure filesystem staging: Linux-only sealed anonymous descriptors.
- Promotion: requires exact-head CI, fresh independent review, and explicit human approval.

## Invariants implemented

1. **Identity and state remain external to workers.** SQLite WAL snapshots exist as an administrative store. Workers cannot write the database directly, and bounded v0.1 rejects every returned `_state` proposal before starting a SQLite write transaction.
2. **Signed capability leases.** A lease binds issuer, exact worker subject, capability, resource scope, validity window, nonce, and metadata.
3. **Trusted worker policy.** Callers choose only registered `module:function` targets and may request only stricter finite budgets. Capability, logical resource, filesystem fields, payload ceiling, and maximum wall-clock/memory/CPU budgets come from the trusted registry.
4. **Bounded payload admission.** The monotonic execution deadline starts before caller payload copying, canonicalization, or hashing. Canonical JSON serialization runs in a killable child and is capped at the policy-owned byte ceiling before any worker launch.
5. **Human STOP and revocation.** Lease validity is checked before execution, after child completion, and immediately before issuing a success receipt. Bounded execution performs no SQLite write transaction or public filesystem mutation, so those uninterruptible finalizers cannot retain authority past STOP.
6. **Pinned Python worker bootstrap.** Python isolated mode ignores ambient `PYTHONPATH`, CWD, and user-site imports, then inserts only the source root derived from the already-loaded trusted runner.
7. **Bounded process execution.** POSIX CPU/address-space limits are applied inside the exec'd worker entry. Workers run in a dedicated process group; timeout terminates descendants and bounds pipe draining even after the group leader exits.
8. **Descriptor-safe filesystem input.** A signed scope is traversed with `O_DIRECTORY|O_NOFOLLOW`. A bounded child copies the authorized regular input directly into a Linux `memfd`, applies write/grow/shrink/seal seals, and only then transfers the anonymous descriptor. No temporary-directory pathname is created or trusted.
9. **Dry-run output isolation.** OMEN dry-run receives an anonymous output descriptor plus a trusted format hint. The worker receives neither caller filesystem path. Any non-dry-run file-producing request is rejected before worker or FFmpeg launch.
10. **Signed execution receipts.** Successful stateless or dry-run work returns a signed receipt containing worker identity, capability/resource, canonical bounded-input hash, output hash, and unchanged pre/post state hashes.
11. **Fail closed.** Invalid/tampered/expired/revoked leases, policy mismatches, unregistered workers, over-budget values, oversized or non-JSON payloads, filesystem escapes, symlinked input components, input aliases, staging faults, worker timeouts, state proposals, and native file-producing requests are rejected.

## Acceptance matrix

| Case | Expected result |
|---|---|
| Valid lease + registered stateless worker | Execute and issue a signed receipt |
| Valid lease + OMEN dry run + allowed paths | Copy input into sealed anonymous storage, evaluate the contract, restore public path labels in the result, and issue a signed receipt |
| Wrong capability, worker subject, or logical resource | Reject |
| Out-of-scope filesystem path or colon-sibling escape | Reject |
| Input symlink component or multi-link input | Reject before worker launch |
| Oversized payload | Reject before worker launch |
| Payload canonicalization exceeds the deadline | Kill the serializer and reject |
| Temporary-directory creation or pathname substitution attempt | No effect; runtime creates no pathname-backed staging directory |
| Mutation attempt against staged input | Reject at the sealed anonymous file |
| Worker returns `_state` | Reject before SQLite BEGIN/COMMIT/ROLLBACK |
| Deadline-bearing administrative state write/commit guard | Reject before transaction finalization |
| Non-dry-run OMEN or any native file-producing request | Reject before worker/FFmpeg launch; no output and no success receipt |
| Public publication helper invoked directly | Reject; `os.replace` cannot execute |
| Worker timeout with descendants | Kill the process group and bound pipe draining |
| Ambient CWD/`PYTHONPATH` shadow | Ignore and execute the pinned package entry |
| Duplicate identical invocations | Distinct signed receipt IDs |
| OMEN command contract | Hard 48 kHz plus EBU R128 `loudnorm`; supported suffix selects codec |
| OMEN 44.1 kHz override | Reject |

`tests/test_sovereign_runtime.py` defines 39 unit/security-contract tests. The suite includes adversarial and fault-injection coverage for payload admission, authority checks, STOP/revocation races, process-group teardown, descriptor traversal, sealed anonymous input, blocked input creation/read/preflight, parent-free timeout recovery, lexical output scope, prohibited state transactions, prohibited native FFmpeg execution, and prohibited public publication.

## OMEN dry-run worker

The registered `sunokiller.omen:mastering_worker` currently supports contract evaluation only. The broker:

1. lexically selects a signed input/output scope;
2. opens and validates the input through no-follow descriptors;
3. copies and seals input inside an anonymous Linux memory-backed file;
4. supplies only inherited `/proc/self/fd/<n>` objects to the worker;
5. builds the 48 kHz EBU R128 command using the original output suffix as a trusted format hint; and
6. restores the caller-visible path labels in the returned dry-run result.

A non-dry-run request fails before the worker starts. The configured FFmpeg pathname is therefore not verified or executed by bounded v0.1. Restoring native execution requires an immutable/opened executable binding and a separately proven output transaction.

Standalone CLI use remains outside this bounded leased runtime:

```bash
omen mix.wav -o master.wav --target-lufs -14 --true-peak -1
```

Standalone operation is not evidence of bounded-runtime authorization, receipt atomicity, deployment, or public delivery.

## Administrative state-store boundary

The SQLite store retains direct administrative snapshot/revocation APIs and optimistic concurrency tests. Calls without an execution deadline may block in SQLite or filesystem finalization. Calls that present a bounded execution deadline are refused before a write transaction because in-process deadline checks cannot interrupt a stalled COMMIT or ROLLBACK. Worker state mutation must remain disabled until transaction finalization and its receipt are supervised as one recoverable protocol.

## Deliberate non-claims

- HMAC-SHA256 is symmetric v0.1 signing, not final Ed25519 hardening.
- The trusted worker registry is static, not a signed manifest/code-digest registry.
- Process isolation is not a container, seccomp policy, VM, or separate service UID.
- Anonymous filesystem staging requires Linux `memfd_create`, sealing, descriptor passing, and `/proc/self/fd`; unsupported hosts fail closed.
- Native FFmpeg execution is disabled in the leased runtime.
- Worker-driven durable state mutation is disabled.
- Public output publication and crash durability are unimplemented.
- Perception, synthesis, stem separation, trained-model quality, physical-device acceptance, deployment, and a full sovereign DAW remain unverified.

## Verification gate

Before merge or promotion:

1. The current PR head must remain unchanged, open, and mergeable.
2. The exact-head `Sovereign Runtime Boundary` workflow must pass on Python 3.11 and 3.12.
3. Independent review must target that exact head; every P1/P2 remains blocking until repaired and re-reviewed.
4. The PR description and this document must match the exact head's reduced authority and non-claims.
5. Explicit human approval must bind the unchanged reviewed SHA. An earlier or unpinned approval does not carry across head mutations.

## Next integration slices

1. Design a killably supervised, recoverable SQLite finalization plus receipt protocol before enabling worker state mutation.
2. Bind native executables to immutable opened objects before enabling FFmpeg.
3. Design and adversarially prove a durable receipt-linked publication transaction.
4. Define `DAW_CAPABILITY_MATRIX-v1`, then register perception, synthesis, and stem separation as separate reviewed leases.
5. Chain receipts so every stage proves the exact prior-stage input hash consumed.
