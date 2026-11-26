import pytest
import torch

from autoresearch import common_search as search
from nanoscigpt.tasks import eleven


@pytest.mark.parametrize("field", list(eleven.FIELDS))
def test_every_field_shares_the_search_space_and_keeps_the_backbone(field):
    default = search.start(field)
    assert set(search.space(field)) == set(default)
    assert search.canonical(field, search.validate(field, default)) == default
    with pytest.raises(ValueError):
        search.validate(field, {**default, "n_layer": 4})
    with pytest.raises(ValueError):
        search.validate(field, {**default, "merge": 3})
    model = search.network(field, merge=2)
    assert len(model.blocks) == 2 and model.head.in_features == 32


def test_merging_joins_neighbouring_tokens():
    symbols = {"tokens": torch.tensor([[2, 3, 4, 0]])}
    merged = search.regroup("dna", symbols, 2)["tokens"]
    assert merged.tolist() == [[2 * 6 + 3, 4 * 6 + 0]]  # a pair of padding symbols stays 0, the padding id
    assert search.regroup("dna", {"tokens": torch.zeros(1, 4, dtype=torch.long)}, 2)["tokens"].eq(0).all()
    patches = {"patches": torch.arange(2 * 16 * 4, dtype=torch.float32).reshape(2, 16, 4)}
    joined = search.regroup("spectrum", patches, 2)["patches"]
    assert joined.shape == (2, 8, 8) and joined[0, 0].tolist() == list(range(8))
    cell = {"elements": torch.ones(1, 5, dtype=torch.long)}
    assert search.regroup("crystal", cell, 2) is cell


def test_settings_without_effect_are_merged():
    default = search.start("protein")
    assert search.canonical("protein", {**default, "mask_ratio": 0.5, "finetune_steps": 400}) == default
    masked = {**default, "objective": "masked", "mask_ratio": 0.5}
    assert search.canonical("protein", masked) == masked
    assert search.canonical("crystal", {**search.start("crystal"), "merge": 2, "ridge_alpha": 10.0}) == search.start("crystal")


def test_verdict_follows_the_written_rule():
    def run(method, repeat, error, chosen_round=3):
        return {"method": method, "repeat": repeat,
                "confirmation": {"A": error, "round": chosen_round, "start_A": 1.0, "guess_mean_A": 2.0}}

    clear = search.verdict([run("model", 1, 0.70), run("model", 2, 0.72), run("random", 1, 0.90), run("random", 2, 0.95)])
    assert (clear["against_start"], clear["against_random"]) == ("好于起点", "好于随机搜索")
    close = search.verdict([run("model", 1, 0.70), run("model", 2, 0.90), run("random", 1, 0.85), run("random", 2, 0.86)])
    assert (close["against_start"], close["against_random"]) == ("好于起点", "和随机搜索分不出高下")
    worse = search.verdict([run("model", 1, 1.05), run("model", 2, 0.90), run("random", 1, 0.80), run("random", 2, 0.82)])
    assert (worse["against_start"], worse["against_random"]) == ("没有稳定好于起点", "没有跑赢随机搜索")
    stuck = search.verdict([run("model", 1, 1.0, 0), run("model", 2, 1.0, 0), run("random", 1, 1.0, 0), run("random", 2, 1.0, 0)])
    assert stuck["against_start"] == "没有找到更好的配置"

