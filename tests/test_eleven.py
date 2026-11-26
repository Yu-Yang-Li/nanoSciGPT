import numpy as np
import pytest

from nanoscigpt.tasks import eleven


@pytest.mark.parametrize("field", ["protein", "dna", "rna", "smiles", "crystal", "text", "spectrum", "structure3d"])
def test_panels_have_inputs_and_task_labels(field, capsys):
    parts, task = eleven.load(field)
    assert len(next(iter(parts["unlabeled"][0].values()))) == 12800
    for group in ("pool", "A"):
        inputs, labels = task[group]
        assert len(next(iter(inputs.values()))) == len(labels)
        assert np.isfinite(labels).all() and labels.std() > 0
    eleven.show_tokens(field, {name: values[0] for name, values in task["A"][0].items()})
    assert "token" in capsys.readouterr().out


def test_symbol_tables_match_vocabulary_sizes():
    for field, spec in eleven.FIELDS.items():
        if spec["kind"] == "symbol":
            assert len(eleven.symbols(field)) == spec["vocab"]

def test_research_search_keeps_the_backbone_and_matches_the_notebook_tokens():
    import pytest
    from autoresearch import eleven_search as search
    raw, labels = search.fields("pool")
    parts, task = eleven.load("field")
    np.testing.assert_array_equal(search.tokenize(raw, search.START), parts["pool"][0]["patches"].numpy())
    np.testing.assert_allclose(labels, task["pool"][1])
    with pytest.raises(ValueError):
        search.validate({**search.START, "n_layer": 4})
    with pytest.raises(ValueError):
        search.validate({**search.START, "space_per_token": 2})


def test_research_search_merges_settings_that_have_no_effect():
    from autoresearch import eleven_search as search
    raw, _ = search.fields("pool")
    by_time = {**search.START, "order": "time"}
    np.testing.assert_array_equal(search.tokenize(raw[:4], by_time), search.tokenize(raw[:4], search.START))
    assert search.canonical(by_time) == search.START
    assert search.canonical({**search.START, "mask_ratio": 0.5, "ridge_alpha": 10.0}) == search.START
    probe = {**search.START, "readout": "probe", "finetune_steps": 400}
    assert search.canonical(probe) == {**search.START, "readout": "probe"}
    masked = {**search.START, "objective": "masked", "mask_ratio": 0.5}
    assert search.canonical(masked) == masked
    assert len(search.DEV_SEEDS[1]) == 3 and len(search.DEV_SEEDS[2]) == 5


def test_stream_orbit_fits_gd1_and_mock_truth_scores_near_data_count():
    from autoresearch import stream_fit, stream_skill
    real = stream_fit.chi2(stream_skill.REFERENCE)[0].sum()
    assert real < 125  # large-budget fit to the 59 GD-1 measurements reached 117
    case = stream_skill.scenario(9001)
    assert 20 < case["truth_chi2"] < 120
    assert stream_fit.chi2([150, 1.5, 0.5, 13, -5, 1, -150])[0].min() == 1e6  # orbit misses the phi1 span
    batch, _ = stream_skill.parse_candidates('{"reason": "", "candidates": [[999, 0.9, -1, 10, -9, -2, -250], [1, 2]]}')
    assert batch.shape == (1, 7) and batch[0, 0] == stream_fit.PRIOR["vc"][1]


def test_equal_budget_stream_fit_requires_ten_finite_parameter_vectors():
    import json
    from autoresearch.stream_skill import REFERENCE, parse_candidates

    rows = [REFERENCE.tolist() for _ in range(10)]
    batch, _ = parse_candidates(json.dumps({"candidates": rows}), exact=True)
    assert batch.shape == (10, 7)
    for invalid in (rows[:-1], rows + rows[:1], [[float("nan")] * 7] + rows[1:]):
        with pytest.raises(ValueError):
            parse_candidates(json.dumps({"candidates": invalid}), exact=True)
