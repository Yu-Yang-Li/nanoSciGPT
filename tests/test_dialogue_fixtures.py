import json

import numpy as np
import pytest

from scripts import prepare_dialogue_fixtures


def test_retest_fixtures_are_portable_and_not_reported_as_real_measurements(tmp_path):
    output = tmp_path / "student-data"
    prepare_dialogue_fixtures.prepare(output)
    assert (output / "student-solubility.csv").is_file()
    with np.load(output / "student-weather.npz") as weather:
        assert weather["train_x"].shape == (80, 4, 16, 16)
        assert weather["train_y"].shape == (80,)
    metadata = json.loads((output / "fixture_manifest.json").read_text())
    assert metadata["measurement_status"] == "synthetic_test_input_not_a_run"
    assert json.loads((output / "student-model" / "metrics.json").read_text()) == {"rmse": 0.42}
    (output / "student-solubility.csv").write_text("student edit")
    with pytest.raises(FileExistsError):
        prepare_dialogue_fixtures.prepare(output)
    assert (output / "student-solubility.csv").read_text() == "student edit"
