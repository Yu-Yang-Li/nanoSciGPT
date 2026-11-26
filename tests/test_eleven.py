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
