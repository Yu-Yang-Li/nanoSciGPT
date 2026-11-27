"""The tutorial notebook's figures must draw for every field, and its archives must load."""
import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nanoscigpt.tasks import eleven, eleven_plots as plots  # noqa: E402

plots.use_chinese_font()  # the notebook does this in its first cell


@pytest.fixture(autouse=True)
def count_figures(monkeypatch):
    shown = []

    def show():
        figure = plt.gcf()
        assert figure._suptitle is not None and figure._suptitle.get_text(), "every figure states its conclusion"
        figure.canvas.draw()  # the layout must fit: this is where a broken figure would fail
        shown.append(len(plt.get_fignums()))
        plt.close("all")

    monkeypatch.setattr(plt, "show", show)
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


def test_headline_wraps_long_conclusions_and_style_is_shared():
    from nanoscigpt import style

    style.use()
    assert plt.rcParams["font.size"] >= 12 and not plt.rcParams["axes.spines.top"]
    figure = plt.figure(figsize=(6, 3))
    style.headline(figure, "结论" * 40)
    lines = figure._suptitle.get_text().split("\n")
    assert len(lines) > 1 and "".join(lines) == "结论" * 40
    assert style.relative(0.9, 1.0) == "低 10%" and style.relative(1.25, 1.0) == "高 25%"
    plt.close("all")


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

    stopped = []
    for field in eleven.FIELDS:
        runs = search.archived(field)
        counts = {method: sorted(r["repeat"] for r in runs if r["method"] == method) for method in ("model", "random", "skill")}
        assert counts["model"] == counts["random"] == counts["skill"] and counts["model"] in ([1, 2], [1, 2, 3, 4])
        for run in runs:
            assert run["trials"][0]["config"] == search.start(field)
            if "stopped_early" in run:  # a run the model could not continue; it keeps the rounds it finished
                stopped.append((field, run["method"], run["repeat"]))
                assert len(run["trials"]) == run["stopped_early"]["rounds_completed"] + 1 < 9
                assert len(search_plots.best_so_far(run)) == 9
            else:
                assert len(run["trials"]) == 9
            assert json.dumps(run).count("SCNET") == 0
    assert stopped == [("crystal", "skill", 1), ("crystal", "skill", 4)]  # docs/search-skill.md explains these two
    search_plots.show_search("field", search.archived("field"))
    search_plots.show_all_fields()
    search_plots.show_skill_transfer()
    assert count_figures == [1, 1, 1]


def test_every_skill_run_used_the_locked_skill_that_never_saw_its_field():
    import hashlib

    from autoresearch import common_search as search, search_skill

    assert sorted(f for fold in search_skill.FOLDS.values() for f in fold) == sorted(eleven.FIELDS)
    for field in eleven.FIELDS:
        skill = search_skill.locked(field)
        digest = hashlib.sha256(skill.encode("utf-8")).hexdigest()
        assert {r["skill_sha256"] for r in search.archived(field) if r["method"] == "skill"} == {digest}
        prompt = search_skill.write_prompt(search_skill.fold_of(field))
        assert f"## {eleven.FIELDS[field]['name']}：" not in prompt and eleven.FIELDS[field]["question"] not in prompt

