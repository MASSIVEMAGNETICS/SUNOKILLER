"""Child-process entry point for the sovereign runtime."""

from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Any, Mapping


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def _apply_resource_limits(limits: Mapping[str, Any]) -> None:
    """Apply trusted parent-validated limits after exec, before worker import.

    Running this inside the already-exec'd child avoids Python `preexec_fn` in
    the multithreaded host. POSIX descendants inherit these limits. Non-POSIX
    hosts retain the trusted parent wall-clock timeout but do not claim these
    process-level CPU/address-space limits.
    """
    if not isinstance(limits, dict):
        raise TypeError("limits must be a JSON object")

    memory_mb = _positive_int(limits.get("max_memory_mb"), "max_memory_mb")
    cpu_seconds = _positive_int(limits.get("max_cpu_seconds"), "max_cpu_seconds")

    if os.name != "posix":
        return

    import resource

    memory_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))


def main() -> None:
    try:
        envelope = json.load(sys.stdin)
        module = str(envelope["module"])
        function = str(envelope["function"])
        payload = envelope["payload"]
        limits = envelope["limits"]
        if not isinstance(payload, dict):
            raise TypeError("payload must be a JSON object")

        # Apply resource bounds before importing the registered worker module,
        # so worker import and all descendants execute inside the policy budget.
        _apply_resource_limits(limits)

        target = getattr(importlib.import_module(module), function)
        result = target(payload)
        if not isinstance(result, dict):
            raise TypeError("worker must return a dict")

        json.dump({"ok": True, "result": result}, sys.stdout, separators=(",", ":"))
    except BaseException as exc:
        json.dump(
            {"ok": False, "error": "{}: {}".format(type(exc).__name__, exc)},
            sys.stdout,
            separators=(",", ":"),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
