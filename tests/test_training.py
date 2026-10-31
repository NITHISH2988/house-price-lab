import json

import pytest

from house_price_lab.train import train


def test_training_persists_reproducible_baseline_comparison(tmp_path):
    first = train(tmp_path / "first", rows=100, seed=7)
    second = train(tmp_path / "second", rows=100, seed=7)

    assert first["cv_mae"] == pytest.approx(second["cv_mae"], abs=1e-8)
    assert first["test_metrics"] == pytest.approx(second["test_metrics"], abs=1e-8)
    assert "mean_baseline" in first["cv_mae"]
    assert (tmp_path / "first" / "model.joblib").is_file()
    metrics = json.loads(
        (tmp_path / "first" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["synthetic"] is True
    assert metrics["seed"] == 7
