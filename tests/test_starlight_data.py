"""Portable classroom entrypoints must preserve real labels and predictions."""
import io
import zipfile
import joblib
import numpy as np
import pytest


def test_parser_preserves_original_labels_and_rejects_fractional_labels():
    from nanoscigpt.starlight_data import parse_split
    rows = np.zeros((3, 1025))
    rows[:, 0] = [3, 1, 2]
    rows[:, 1] = [10, 20, 30]
    stream = io.StringIO()
    np.savetxt(stream, rows)
    x, y = parse_split(stream.getvalue().encode())
    assert y.tolist() == [3, 1, 2]
    assert x[:, 0].tolist() == [10, 20, 30]
    rows[0, 0] = 1.5
    stream = io.StringIO()
    np.savetxt(stream, rows)
    with pytest.raises(ValueError, match='labels'):
        parse_split(stream.getvalue().encode())


def test_wrong_archive_is_rejected_before_output_creation(tmp_path):
    from nanoscigpt.starlight_data import prepare
    archive = tmp_path / 'fake.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('StarLightCurves_TRAIN.txt', '1 2 3')
    with pytest.raises(ValueError, match='SHA256'):
        prepare(archive, tmp_path / 'data')
    assert not (tmp_path / 'data').exists()


def test_baseline_reloads_and_never_trains_on_validation_or_test(tmp_path):
    from nanoscigpt.starlight_data import baseline
    source = tmp_path / 'data.npz'
    x = np.arange(18 * 16).reshape(18, 16).astype('float32')
    y = np.tile([1, 2, 3], 6)
    np.savez(source, x_train=x, y_train=y, fit_indices=np.arange(12),
             validation_indices=np.arange(12, 18), x_test=x[:6] + .5, y_test=y[:6])
    out = tmp_path / 'baseline'
    result = baseline(source, out)
    assert result['fit_samples'] == 12
    assert result['validation_samples'] == 6
    model = joblib.load(out / 'random_forest.joblib')
    assert all(tree.tree_.n_node_samples[0] <= 12 for tree in model.estimators_)
    with np.load(out / 'predictions.npz') as p:
        np.testing.assert_array_equal(p['random_forest'], model.predict(x[:6] + .5))
        assert p['y_true'].tolist() == [1, 2, 3, 1, 2, 3]
    with pytest.raises(FileExistsError):
        baseline(source, out)
