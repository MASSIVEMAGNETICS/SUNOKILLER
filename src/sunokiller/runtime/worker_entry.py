"""Child-process entry point for the sovereign runtime."""

from __future__ import annotations

import importlib
import json
import sys


def main() -> None:
    try:
        envelope = json.load(sys.stdin)
        module = str(envelope["module"])
        function = str(envelope["function"])
        payload = envelope["payload"]
        if not isinstance(payload, dict):
            raise TypeError("payload must be a JSON object")

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
