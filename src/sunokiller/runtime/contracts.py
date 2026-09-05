"""Zero-trust capability lease and execution receipt contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import hmac
import json
from pathlib import Path
import secrets
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


class LeaseError(RuntimeError):
    """Base class for lease validation failures."""


class InvalidSignature(LeaseError):
    pass


class LeaseNotYetValid(LeaseError):
    pass


class LeaseExpired(LeaseError):
    pass


class LeaseRevoked(LeaseError):
    pass


class CapabilityDenied(LeaseError):
    pass


class ResourceDenied(LeaseError):
    pass


def canonical_json(value: Any) -> str:
    """Stable JSON encoding used for signatures and hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_json(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _logical_scope_match(resource: str, allowed_scope: str) -> bool:
    """Match logical resource namespaces such as catalog/masters or audio:master."""
    if allowed_scope == "*":
        return True
    if resource == allowed_scope:
        return True
    return (
        resource.startswith(allowed_scope.rstrip("/") + "/")
        or resource.startswith(allowed_scope.rstrip(":") + ":")
    )


def _filesystem_scope_match(resource: str, allowed_scope: str) -> bool:
    """Filesystem-aware containment; never interpret ':' as a path hierarchy.

    Filesystem scopes must be absolute paths (or '*'). Both resource and scope
    are resolved before containment. This prevents a sibling such as
    `/srv/allowed:outside/file.wav` from matching `/srv/allowed`.
    """
    if allowed_scope == "*":
        return True

    scope_path = Path(allowed_scope).expanduser()
    resource_path = Path(resource).expanduser()
    if not scope_path.is_absolute() or not resource_path.is_absolute():
        return False

    scope_path = scope_path.resolve()
    resource_path = resource_path.resolve()
    try:
        resource_path.relative_to(scope_path)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class CapabilityLease:
    lease_id: str
    issuer: str
    subject: str
    capabilities: Tuple[str, ...]
    resource_scopes: Tuple[str, ...]
    not_before: int
    expires_at: int
    nonce: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    signature: str = ""

    def unsigned_payload(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("signature", None)
        data["metadata"] = dict(self.metadata)
        return data

    def with_signature(self, signature: str) -> "CapabilityLease":
        return CapabilityLease(**dict(self.unsigned_payload(), signature=signature))


@dataclass(frozen=True)
class ExecutionReceipt:
    execution_id: str
    lease_id: str
    worker: str
    capability: str
    resource: str
    started_at: int
    finished_at: int
    pre_state_hash: str
    post_state_hash: str
    input_hash: str
    output_hash: str
    status: str
    error: Optional[str] = None
    signature: str = ""

    def unsigned_payload(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("signature", None)
        return data

    def with_signature(self, signature: str) -> "ExecutionReceipt":
        return ExecutionReceipt(**dict(self.unsigned_payload(), signature=signature))


class HMACAuthority:
    """Dependency-free signing authority for the first deployable slice.

    The signing secret must remain outside repository state. The lease/receipt
    schema is deliberately algorithm-agnostic so HMAC can later be replaced by
    Ed25519 without changing the execution contract.
    """

    def __init__(self, secret: bytes, issuer: str = "ETHICA_AUTHORITY") -> None:
        if len(secret) < 32:
            raise ValueError("lease signing secret must be at least 32 bytes")
        self._secret = secret
        self.issuer = issuer

    @classmethod
    def from_hex(cls, secret_hex: str, issuer: str = "ETHICA_AUTHORITY") -> "HMACAuthority":
        return cls(bytes.fromhex(secret_hex), issuer=issuer)

    @staticmethod
    def new_secret_hex() -> str:
        return secrets.token_hex(32)

    def _sign_payload(self, payload: Mapping[str, Any]) -> str:
        return hmac.new(
            self._secret,
            canonical_json(dict(payload)).encode("utf-8"),
            sha256,
        ).hexdigest()

    def issue_lease(
        self,
        *,
        subject: str,
        capabilities: Sequence[str],
        resource_scopes: Sequence[str],
        ttl_seconds: int = 300,
        not_before: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CapabilityLease:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        now = int(time.time()) if not_before is None else int(not_before)
        lease = CapabilityLease(
            lease_id="LEASE-" + secrets.token_hex(12),
            issuer=self.issuer,
            subject=subject,
            capabilities=tuple(sorted(set(capabilities))),
            resource_scopes=tuple(sorted(set(resource_scopes))),
            not_before=now,
            expires_at=now + int(ttl_seconds),
            nonce=secrets.token_hex(16),
            metadata=dict(metadata or {}),
        )
        return lease.with_signature(self._sign_payload(lease.unsigned_payload()))

    def verify_lease(
        self,
        lease: CapabilityLease,
        *,
        required_capability: str,
        resource: str,
        now: Optional[int] = None,
        revoked_ids: Iterable[str] = (),
        resource_kind: str = "logical",
    ) -> None:
        expected = self._sign_payload(lease.unsigned_payload())
        if not hmac.compare_digest(expected, lease.signature):
            raise InvalidSignature("lease signature mismatch")
        if lease.issuer != self.issuer:
            raise InvalidSignature("unexpected lease issuer")

        current = int(time.time()) if now is None else int(now)
        if current < lease.not_before:
            raise LeaseNotYetValid("lease is not yet valid")
        if current >= lease.expires_at:
            raise LeaseExpired("lease has expired")
        if lease.lease_id in set(revoked_ids):
            raise LeaseRevoked("lease has been revoked")
        if required_capability not in lease.capabilities:
            raise CapabilityDenied(required_capability)

        if resource_kind == "logical":
            matcher = _logical_scope_match
        elif resource_kind == "filesystem":
            matcher = _filesystem_scope_match
        else:
            raise ValueError("unsupported resource_kind: {}".format(resource_kind))

        if not any(matcher(resource, scope) for scope in lease.resource_scopes):
            raise ResourceDenied(resource)

    def sign_receipt(self, receipt: ExecutionReceipt) -> ExecutionReceipt:
        return receipt.with_signature(self._sign_payload(receipt.unsigned_payload()))

    def verify_receipt(self, receipt: ExecutionReceipt) -> None:
        expected = self._sign_payload(receipt.unsigned_payload())
        if not hmac.compare_digest(expected, receipt.signature):
            raise InvalidSignature("receipt signature mismatch")
