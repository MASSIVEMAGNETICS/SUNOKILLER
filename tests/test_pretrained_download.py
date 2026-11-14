"""Offline regression tests for SUNOKILLER's pretrained download cache."""

import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock


class _FakeTqdm:
    def __init__(self, *args, **kwargs):
        self.n = 0
        self.total = kwargs.get("total")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, amount):
        self.n += amount


def _load_pretrained_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.nn = types.SimpleNamespace(Module=object)
    fake_torch.load = mock.Mock()
    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = _FakeTqdm

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "sunokiller"
        / "pretrained.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_sunokiller_pretrained_download_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"torch": fake_torch, "tqdm": fake_tqdm}):
        spec.loader.exec_module(module)
    return module, fake_torch


class PretrainedDownloadCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.fake_torch = _load_pretrained_module()

    def test_interrupted_download_preserves_existing_cache_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.pt"
            target.write_bytes(b"known-good")

            def partial_then_fail(url, filename, reporthook=None):
                Path(filename).write_bytes(b"partial")
                raise OSError("simulated interrupted transfer")

            with mock.patch.object(
                self.module, "urlretrieve", side_effect=partial_then_fail
            ):
                with self.assertRaises(OSError):
                    self.module.download_file("https://example.test/model.pt", target)

            self.assertEqual(target.read_bytes(), b"known-good")
            self.assertEqual(list(root.glob(".model.pt.*.part")), [])

    def test_empty_download_is_rejected_and_temporary_file_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.pt"

            def write_empty(url, filename, reporthook=None):
                Path(filename).write_bytes(b"")

            with mock.patch.object(self.module, "urlretrieve", side_effect=write_empty):
                with self.assertRaises(OSError):
                    self.module.download_file("https://example.test/model.pt", target)

            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".model.pt.*.part")), [])

    def test_successful_download_replaces_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.pt"
            target.write_bytes(b"old")

            def write_new(url, filename, reporthook=None):
                Path(filename).write_bytes(b"new-weights")

            with mock.patch.object(self.module, "urlretrieve", side_effect=write_new):
                self.module.download_file("https://example.test/model.pt", target)

            self.assertEqual(target.read_bytes(), b"new-weights")
            self.assertEqual(list(root.glob(".model.pt.*.part")), [])

    def test_empty_legacy_placeholder_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            model_dir = cache / "vocos-24khz"
            model_dir.mkdir()
            model_path = model_dir / "model.pt"
            model_path.write_bytes(b"")

            def write_artifact(url, filename, reporthook=None):
                path = Path(filename)
                if url.endswith("model.pt"):
                    path.write_bytes(b"valid-weights")
                else:
                    path.write_text("{}", encoding="utf-8")

            with mock.patch.object(self.module, "get_cache_dir", return_value=cache):
                with mock.patch.object(self.module, "urlretrieve", side_effect=write_artifact):
                    paths = self.module.download_pretrained_model("vocos-24khz")

            self.assertEqual(paths["model"].read_bytes(), b"valid-weights")
            self.assertTrue(paths["config"].is_file())

    def test_failed_first_download_leaves_no_placeholder_and_later_call_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            model_calls = 0

            def fail_once_then_succeed(url, filename, reporthook=None):
                nonlocal model_calls
                path = Path(filename)
                if url.endswith("model.pt"):
                    model_calls += 1
                    if model_calls == 1:
                        path.write_bytes(b"partial")
                        raise OSError("simulated network interruption")
                    path.write_bytes(b"valid-weights")
                else:
                    path.write_text("{}", encoding="utf-8")

            with mock.patch.object(self.module, "get_cache_dir", return_value=cache):
                with mock.patch.object(self.module, "urlretrieve", side_effect=fail_once_then_succeed):
                    with redirect_stdout(StringIO()):
                        first = self.module.download_pretrained_model("vocos-24khz")
                    self.assertFalse(first["model"].exists())
                    with redirect_stdout(StringIO()):
                        second = self.module.download_pretrained_model("vocos-24khz")

            self.assertEqual(model_calls, 2)
            self.assertEqual(second["model"].read_bytes(), b"valid-weights")

    def test_failed_forced_refresh_preserves_existing_model_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            model_dir = cache / "vocos-24khz"
            model_dir.mkdir()
            model_path = model_dir / "model.pt"
            config_path = model_dir / "config.json"
            model_path.write_bytes(b"old-valid-model")
            config_path.write_bytes(b'{"old": true}')

            def fail_after_partial(url, filename, reporthook=None):
                Path(filename).write_bytes(b"partial")
                raise OSError("simulated refresh failure")

            with mock.patch.object(self.module, "get_cache_dir", return_value=cache):
                with mock.patch.object(self.module, "urlretrieve", side_effect=fail_after_partial):
                    with redirect_stdout(StringIO()):
                        paths = self.module.download_pretrained_model(
                            "vocos-24khz",
                            force_download=True,
                        )

            self.assertEqual(paths["model"].read_bytes(), b"old-valid-model")
            self.assertEqual(paths["config"].read_bytes(), b'{"old": true}')
            self.assertEqual(list(model_dir.glob(".*.part")), [])

    def test_missing_weights_keep_explicit_random_initialization_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)

            def fail_transfer(url, filename, reporthook=None):
                Path(filename).write_bytes(b"partial")
                raise OSError("offline")

            model = object()
            with mock.patch.object(self.module, "get_cache_dir", return_value=cache):
                with mock.patch.object(self.module, "urlretrieve", side_effect=fail_transfer):
                    with redirect_stdout(StringIO()):
                        result = self.module.load_pretrained_weights(model, "vocos-24khz")

            self.assertIs(result, model)
            self.fake_torch.load.assert_not_called()
            self.assertFalse((cache / "vocos-24khz" / "model.pt").exists())


if __name__ == "__main__":
    unittest.main()
