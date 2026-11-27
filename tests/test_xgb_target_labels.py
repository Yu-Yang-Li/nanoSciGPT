"""Fit the distributed template on the original, string-labelled penguin CSV."""
import subprocess
import sys
from pathlib import Path

import pytest


def test_xgb_template_roundtrips_species_labels(tmp_path):
    pytest.importorskip("optuna")
    pytest.importorskip("xgboost")
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    root = Path(__file__).resolve().parents[1]
    csv = root / "data/course/palmer_penguins_morphology.csv"
    template = root / "skills/research-baseline-builder/templates/baseline_xgboost_optuna.py"
    subprocess.run([sys.executable, str(template), "--csv", str(csv), "--target", "species",
                    "--n_trials", "1"], cwd=tmp_path, check=True, capture_output=True)
    import json
    bundle = joblib.load(tmp_path / "xgb_optuna_model.joblib")
    frame = pd.read_csv(csv)
    X = pd.get_dummies(frame.drop(columns="species"), drop_first=False)
    _, test, _, labels = train_test_split(X, frame.species, test_size=0.2, stratify=frame.species, random_state=42)
    predictions = bundle["label_encoder"].inverse_transform(bundle["model"].predict(test))
    assert set(bundle["label_encoder"].classes_) == set(frame.species)
    assert accuracy_score(labels, predictions) == json.loads((tmp_path / "metrics.json").read_text())["test_accuracy"]

