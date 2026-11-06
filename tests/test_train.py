"""The loss curve comes from mlx-lm's training callback (no log parsing)."""

import importlib.util

import pytest

from distill.runinfo import REPO_ROOT

pytest.importorskip("mlx_lm")
spec = importlib.util.spec_from_file_location("train", REPO_ROOT / "scripts/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def test_loss_recorder_keeps_every_report_as_floats():
    rec = train.LossRecorder()
    rec.on_val_loss_report({"iteration": 0, "val_loss": 1.098, "val_time": 62.5})
    rec.on_train_loss_report(
        {"iteration": 10, "train_loss": 0.9, "peak_memory": 9.1, "trained_tokens": 26600}
    )
    assert rec.val == [{"iteration": 0.0, "val_loss": 1.098, "val_time": 62.5}]
    assert rec.train[0]["train_loss"] == 0.9 and rec.train[0]["peak_memory"] == 9.1
