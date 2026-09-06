"""OMEN mastering execution harness.

The mastering stage is independent from neural synthesis and stem-separation
workers. A verified stem-separation worker can feed OMEN through the same
capability boundary; this harness produces 48 kHz commercial masters from a
mix or pre-rendered stem mix.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List


class OmenError(RuntimeError):
    pass


MASTER_SAMPLE_RATE = 48000

_FORMAT_CODECS = {
    ".wav": ["-c:a", "pcm_s24le"],
    ".flac": ["-c:a", "flac"],
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
}


def _require_master_sample_rate(sample_rate: int) -> None:
    if int(sample_rate) != MASTER_SAMPLE_RATE:
        raise OmenError(
            "OMEN mastering contract requires {} Hz, got {}".format(
                MASTER_SAMPLE_RATE, sample_rate
            )
        )


def build_master_command(
    *,
    input_path: str,
    output_path: str,
    target_lufs: float = -14.0,
    true_peak: float = -1.0,
    loudness_range: float = 11.0,
    sample_rate: int = MASTER_SAMPLE_RATE,
) -> List[str]:
    _require_master_sample_rate(sample_rate)
    out = Path(output_path)
    codec = _FORMAT_CODECS.get(out.suffix.lower())
    if codec is None:
        raise OmenError("unsupported output format: {}".format(out.suffix))
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(MASTER_SAMPLE_RATE),
        "-af",
        "loudnorm=I={}:TP={}:LRA={}".format(
            target_lufs, true_peak, loudness_range
        ),
        *codec,
        str(output_path),
    ]


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def master_file(
    *,
    input_path: str,
    output_path: str,
    target_lufs: float = -14.0,
    true_peak: float = -1.0,
    loudness_range: float = 11.0,
    sample_rate: int = MASTER_SAMPLE_RATE,
    timeout_seconds: int = 900,
    dry_run: bool = False,
) -> Dict[str, Any]:
    _require_master_sample_rate(sample_rate)
    source = Path(input_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()

    if not source.is_file():
        raise OmenError("input does not exist: {}".format(source))
    target.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_master_command(
        input_path=str(source),
        output_path=str(target),
        target_lufs=target_lufs,
        true_peak=true_peak,
        loudness_range=loudness_range,
        sample_rate=MASTER_SAMPLE_RATE,
    )

    if dry_run:
        return {
            "status": "DRY_RUN",
            "command": cmd,
            "input": str(source),
            "output": str(target),
            "sample_rate": MASTER_SAMPLE_RATE,
        }

    if shutil.which("ffmpeg") is None:
        raise OmenError("ffmpeg is required for OMEN mastering")

    subprocess.run(cmd, check=True, timeout=timeout_seconds)
    if not target.is_file() or target.stat().st_size == 0:
        raise OmenError("ffmpeg completed without a usable output")

    return {
        "status": "MASTERED",
        "input": str(source),
        "output": str(target),
        "sample_rate": MASTER_SAMPLE_RATE,
        "target_lufs": target_lufs,
        "true_peak": true_peak,
        "loudness_range": loudness_range,
        "sha256": _file_sha256(target),
        "bytes": target.stat().st_size,
    }


def mastering_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Capability-runner entry point."""
    return master_file(
        input_path=payload["input_path"],
        output_path=payload["output_path"],
        target_lufs=float(payload.get("target_lufs", -14.0)),
        true_peak=float(payload.get("true_peak", -1.0)),
        loudness_range=float(payload.get("loudness_range", 11.0)),
        sample_rate=int(payload.get("sample_rate", MASTER_SAMPLE_RATE)),
        timeout_seconds=int(payload.get("timeout_seconds", 900)),
        dry_run=bool(payload.get("dry_run", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OMEN - 48 kHz commercial mastering harness")
    parser.add_argument("input")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--target-lufs", type=float, default=-14.0)
    parser.add_argument("--true-peak", type=float, default=-1.0)
    parser.add_argument("--lra", type=float, default=11.0)
    parser.add_argument("--sample-rate", type=int, choices=[MASTER_SAMPLE_RATE], default=MASTER_SAMPLE_RATE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = master_file(
        input_path=args.input,
        output_path=args.output,
        target_lufs=args.target_lufs,
        true_peak=args.true_peak,
        loudness_range=args.lra,
        sample_rate=args.sample_rate,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
