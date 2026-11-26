"""The tutorial notebook's figures must draw for every field, and its archives must load."""
import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nanoscigpt.tasks import eleven, eleven_plots as plots  # noqa: E402


@pytest.fixture(autouse=True)
def count_figures(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: (shown.append(len(plt.get_fignums())), plt.close("all")))
    return shown


@pytest.mark.parametrize("field", list(eleven.FIELDS))
def test_data_token_and_model_figures_draw(field, count_figures):
    raw_parts, raw_task = eleven.load(field)
    plots.show_samples(field, raw_task)
    plots.show_tokens(field, raw_task)
    parts, task = eleven.normalize(field, raw_parts, raw_task, 53)
    plots.show_normalization(field, raw_task, task)
    plots.show_model([field, "protein"])
    assert count_figures == [1, 1, 1, 1]


def test_training_and_evaluation_figures_draw(count_figures):
    truth = np.linspace(0, 1, 20)
    plots.show_training({200: [1.0, 0.5, 0.4], 800: [1.0, 0.6, 0.3, 0.2]}, {200: 0.45, 800: 0.25})
    plots.show_predictions(truth, {"不预训练": truth[::-1], "800 个对象预训练": truth + 0.1}, 0.25, "标签")
    plots.show_scaling({200: 0.45, 800: 0.25}, {0: 0.3, 200: 0.28, 800: 0.2}, 0.25, "标题", "标签")
    assert count_figures == [1, 1, 1]


def test_readout_can_be_switched_and_predictions_are_returned():
    parts, task = eleven.load("crystal")
    parts, task = eleven.normalize("crystal", parts, task, 53)
    model = eleven.pretrain("crystal", parts, 0, 53)
    predictions = []
    error, guess = eleven.downstream("crystal", model, task, 53, predictions=predictions, readout="probe")
    assert len(predictions) == len(task["A"][1])
    assert error == pytest.approx(float(np.abs(np.array(predictions) - task["A"][1]).mean()))
    assert guess > 0


def test_stream_archive_matches_the_published_blind_scores(count_figures):
    from autoresearch import stream_plots

    runs = stream_plots.archive()
    means = [np.mean([fit["gap"] for fit in r["blind"].values()]) for r in runs["rounds"]]
    assert np.round(means, 2).tolist() == [2.83, 4.05, 2.04, 2.31, 1.81, 1.64, 1.89, 1.99, 2.18]  # docs/stream-skill.md
    assert len(stream_plots.skill_text(8)) == runs["rounds"][8]["skill_chars"]
    stream_plots.show_orbits()
    stream_plots.show_one_fit()
    stream_plots.show_skill_changes()
    stream_plots.show_blind_scores()
    assert count_figures == [1, 1, 1, 1, 1]


def test_search_archive_covers_every_field(count_figures):
    from autoresearch import common_search as search, search_plots

    for field in eleven.FIELDS:
        runs = search.archived(field)
        assert sorted((r["method"], r["repeat"]) for r in runs) == [("model", 1), ("model", 2), ("random", 1), ("random", 2)]
        for run in runs:
            assert len(run["trials"]) == 9 and run["trials"][0]["config"] == search.start(field)
            assert json.dumps(run).count("SCNET") == 0
    search_plots.show_search("field", search.archived("field"))
    search_plots.show_all_fields()
    assert count_figures == [1, 1]

