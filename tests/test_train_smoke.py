"""训练脚本端到端冒烟测试：小数据集 + fake loader + 1 epoch。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "dualstream" / "train_dualstream.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("train_dualstream", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fake_loader(_path: Path) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)


def _make_dataset(root: Path) -> None:
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        (root / split / "labels_attr").mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text("train: train/images\nval: val/images\nnc: 1\n", encoding="utf-8")
    for i in range(4):
        for split in ("train", "val"):
            (root / split / "images" / f"{i}.jpg").write_bytes(b"")
            (root / split / "labels" / f"{i}.txt").write_text("0 0.5 0.5 0.4 0.3\n", encoding="utf-8")
            color = "red" if i % 2 == 0 else "blue"
            pattern = "one" if i % 2 == 0 else "sentry"
            (root / split / "labels_attr" / f"{i}.txt").write_text(
                f"0 0.5 0.5 0.4 0.3 {color} {pattern}\n", encoding="utf-8"
            )


def test_training_smoke(tmp_path: Path):
    mod = _load_script()
    ds_root = tmp_path / "ds"
    _make_dataset(ds_root)
    out = tmp_path / "runs"
    cfg = mod.DualStreamConfig(y_size=64, c_size=32)
    rc = mod.train(cfg, str(ds_root), epochs=1, device="cpu", batch_size=2, out_dir=str(out), image_loader=_fake_loader)
    assert rc == 0
    assert (out / "last.pt").is_file()
    assert (out / "best.pt").is_file()
