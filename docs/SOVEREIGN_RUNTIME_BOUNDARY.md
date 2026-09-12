# Sovereign Runtime Boundary v0.1

This branch implements a deliberately narrow, local-first execution candidate for Victor / SUNOKILLER. It is a successor boundary, not proof of a complete DAW, deployment, mastering delivery, or autonomous authority.

## Current status

- Implementation: built but partial.
- Public output publication: disabled.
- Native FFmpeg execution inside the bounded runtime: disabled.
- Worker-proposed canonical state mutation: disabled.
- Accepted file-producing mode: OMEN contract dry run only.
- Secure filesystem staging: Linux-only empty sealed dry-run descriptors; input bytes remain outside the worker.
- Promotion: requires exact-head CI, fresh independent review, and explicit human approval.

## Invariants implemented

1. **Identity and state remain external to workers.** SQLite WAL snapshots exist as an administrative store. Workers cannot write the database directly, and bounded v0.1 rejects every returned `_state` proposal before starting a SQLite write transaction.
2. **Signed capability leases.** A lease binds issuer, exact worker subject, capability, resource scope, validity window, nonce, and metadata.
3. **Trusted worker policy.** Callers choose only registered `module:function` targets and may request only stricter finite budgets. Capability, logical resource, filesystem fields, payload ceiling, and maximum wall-clock/memory/CPU budgets come from the trusted registry.
4. **Bounded payload and lease admission.** The monotonic execution deadline starts before caller payload copying, canonicalization, or hashing. A non-allocating exact-built-in structural preflight rejects over-limit payloads before socket/fork setup and repeats inside the serializer child. Signed lease representations use the same 64 KiB ceiling before copying or hashing, and forged leases cannot trigger state reads.
5. **Human STOP and revocation.** Deadline-bearing state-hash and per-lease revocation reads run in killable helper processes bound to the absolute device/inode identity opened at store construction; relative-path changes cannot redirect them, and in-memory stores fail closed for bounded execution. The lease-holding parent performs no synchronous SQLite I/O. Lease validity is checked before execution, after child completion, and immediately before issuing a success receipt.
6. **Pinned Python worker bootstrap.** Python isolated mode ignores ambient `PYTHONPATH`, CWD, and user-site imports, then inserts only the source root derived from the already-loaded trusted runner.
7. **Bounded process execution.** POSIX CPU/address-space limits are applied inside the exec'd worker entry. Workers run in a dedicated process group; timeout terminates descendants and bounds pipe draining even after the group leader exits.
8. **Descriptor-safe dry-run input.** A signed scope is traversed with `O_DIRECTORY|O_NOFOLLOW`, and a bounded child validates the authorized object as a single-link regular file within the trusted size ceiling. Because native execution is disabled and contract evaluation does not consume audio, the child transfers only an empty sealed Linux `memfd` token. No source byte, temporary pathname, or uncharged shmem copy enters the worker.
9. **Dry-run output isolation.** OMEN dry-run receives an anonymous output descriptor plus a trusted format hint. The worker receives neither caller filesystem path. Any non-dry-run file-producing request is rejected before worker or FFmpeg launch.
10. **Signed execution receipts.** Successful stateless or dry-run work returns a signed receipt containing worker identity, capability/resource, canonical bounded-input hash, output hash, and unchanged pre/post state hashes.
11. **Fail closed.** Invalid/tampered/expired/revoked leases, policy mismatches, unregistered workers, over-budget values, oversized or non-JSON payloads, filesystem escapes, symlinked input components, input aliases, staging faults, worker timeouts, state proposals, and native file-producing requests are rejected.

## Acceptance matrix

| Case | Expected result |
|---|---|
| Valid lease + registered stateless worker | Execute and issue a signed receipt |
| Valid lease + OMEN dry run + allowed paths | Validate input metadata, provide an empty sealed contract token, recursively restore public path labels, and issue a signed receipt |
| Wrong capability, worker subject, or logical resource | Reject |
| Out-of-scope filesystem path or colon-sibling escape | Reject |
| Input symlink component or multi-link input | Reject before worker launch |
| Oversized payload | Reject before worker launch |
| Oversized forged lease | Reject before JSON copying/hashing or SQLite access |
| Oversized built-in payload | Reject before socket/fork setup |
| Payload canonicalization exceeds the deadline | Kill the serializer and reject |
| Temporary-directory creation or pathname substitution attempt | No effect; runtime creates no pathname-backed staging directory |
| Input-read or mutation attempt inside the dry-run worker | No source bytes are present; growth/write is rejected by the sealed token |
| Deadline-bearing SQLite read stalls | Kill the read helper and reject without delaying the lease-holding parent |
| CWD changes or state store has no stable file identity | Continue against the originally opened database identity or fail closed |
| Descriptor numbers differ between identical dry runs | Restore nested path labels and produce the same deterministic output hash |
| Worker returns `_state` | Reject before SQLite BEGIN/COMMIT/ROLLBACK |
| Deadline-bearing administrative state write/commit guard | Reject before transaction finalization |
| Non-dry-run OMEN or any native file-producing request | Reject before worker/FFmpeg launch; no output and no success receipt |
| Public publication helper invoked directly | Reject; `os.replace` cannot execute |
| Worker timeout with descendants | Kill the process group and bound pipe draining |
| Ambient CWD/`PYTHONPATH` shadow | Ignore and execute the pinned package entry |
| Duplicate identical invocations | Distinct signed receipt IDs |
| OMEN command contract | Hard 48 kHz plus EBU R128 `loudnorm`; supported suffix selects codec |
| OMEN 44.1 kHz override | Reject |

`tests/test_sovereign_runtime.py` defines 45 unit/security-contract tests. The suite includes adversarial and fault-injection coverage for pre-fork payload rejection, bounded forged-lease verification, database-identity binding, killably supervised SQLite reads, authority checks, STOP/revocation races, process-group teardown, descriptor traversal, empty sealed dry-run tokens, parent-free timeout recovery, deterministic nested path restoration, lexical output scope, prohibited state transactions, prohibited native FFmpeg execution, and prohibited public publication.

## OMEN dry-run worker

The registered `sunokiller.omen:mastering_worker` currently supports contract evaluation only. The broker:

1. lexically selects a signed input/output scope;
2. opens and validates input metadata through no-follow descriptors;
3. creates an empty sealed descriptor token without copying protected input bytes;
4. supplies only inherited `/proc/self/fd/<n>` objects to the worker;
5. builds the 48 kHz EBU R128 command using the original output suffix as a trusted format hint; and
6. recursively restores caller-visible path labels before hashing the returned dry-run result.

A non-dry-run request fails before the worker starts. The configured FFmpeg pathname is therefore not verified or executed by bounded v0.1. Restoring native execution requires an immutable/opened executable binding and a separately proven output transaction.

Standalone CLI use remains outside this bounded leased runtime:

```bash
omen mix.wav -o master.wav --target-lufs -14 --true-peak -1
```

Standalone operation is not evidence of bounded-runtime authorization, receipt atomicity, deployment, or public delivery.

## Administrative state-store boundary

The SQLite store retains direct administrative snapshot/revocation APIs and optimistic concurrency tests. Calls without an execution deadline may block in SQLite or filesystem finalization. A file-backed store normalizes its path at construction and records the opened database device/inode; every helper reconnect must match that identity. Bounded execution reads only small state hashes and single-lease revocation facts in deadline-supervised helper processes; in-memory stores, deadline-bearing full-state loads, and revocation enumeration fail closed. Deadline-bearing writes are refused before a transaction because in-process checks cannot interrupt a stalled COMMIT or ROLLBACK. Worker state mutation must remain disabled until transaction finalization and its receipt are supervised as one recoverable protocol.

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
