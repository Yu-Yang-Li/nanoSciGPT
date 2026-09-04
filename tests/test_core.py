import json
import numpy as np
import subprocess
import sys
import torch
import types

from nanoscigpt.core.gpt import GPT, GPTConfig
from nanoscigpt.core.tokenizer import CharTokenizer


def test_tokenizer_roundtrip():
    tok = CharTokenizer(set("ACGT"))
    s = "ACGTACGT"
    ids = tok.encode(s)
    assert tok.decode(ids) == s


def test_gpt_forward_stream():
    cfg = GPTConfig(vocab_size=20, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)
    x = torch.randint(0, 20, (2, 16))
    y = torch.randint(0, 20, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 20)
    assert loss.item() > 0


def test_gpt_forward_padding():
    cfg = GPTConfig(vocab_size=20, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)
    x = torch.randint(0, 20, (2, 16))
    y = torch.randint(0, 20, (2, 16))
    pad = torch.zeros(2, 16, dtype=torch.bool)
    pad[0, 8:] = True
    logits, loss = model(x, y, pad)
    assert loss.item() > 0


def test_independent_long_sequences_use_more_than_the_prefix(tmp_path):
    from nanoscigpt.core.dataset import IndependentSequenceDataset

    data_dir = tmp_path / "protein"
    data_dir.mkdir()
    sequence = np.arange(1, 101, dtype=np.uint16)
    values = np.empty(1, dtype=object)
    values[0] = sequence
    np.save(data_dir / "train_seqs.npy", values, allow_pickle=True)
    (data_dir / "meta.json").write_text(
        json.dumps({"vocab_size": 101, "mode": "independent", "pad_id": 0}),
        encoding="utf-8",
    )
    dataset = IndependentSequenceDataset(data_dir, "train")
    np.random.seed(7)

    observed_starts = {
        int(dataset.get_batch(batch_size=1, block_size=8, device="cpu")[0][0, 0])
        for _ in range(12)
    }

    assert len(observed_starts) > 1
    assert max(observed_starts) > 1


def test_protein_prepare_keeps_long_canonical_sequences_and_rejects_unknown_residues(tmp_path):
    from nanoscigpt.domains.protein.prepare import parse_fasta_sequences

    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(
        ">long\n" + "A" * 200 + "\n>unknown\nACDX\n>short\nMKWV\n",
        encoding="utf-8",
    )

    sequences, rejected = parse_fasta_sequences(fasta, max_len=0)

    assert [len(sequence) for sequence in sequences] == [200, 4]
    assert rejected == 1


def test_protein_teaching_property_ignores_sequence_control_tokens():
    from nanoscigpt.tasks.downstream_demo import composition_fraction

    sequence = [2, 3, 0, 1]
    assert composition_fraction(sequence, target_ids={2}, ignored_ids={0, 1}) == 0.5


def test_spatial_patch_adapter_preserves_patch_grid():
    from nanoscigpt.scientific.adapters import patchify_2d

    values = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    tokens = patchify_2d(values, patch_size=4)

    assert tokens.shape == (2, 4, 48)


def test_structure_distance_tokens_ignore_rotation_and_translation():
    from nanoscigpt.scientific.adapters import pairwise_distance_tokens

    points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moved = points @ rotation.T + torch.tensor([[[4.0, -3.0, 2.0]]])

    assert torch.allclose(
        pairwise_distance_tokens(points), pairwise_distance_tokens(moved), atol=1e-6
    )


def test_periodic_distances_use_the_shortest_unit_cell_image():
    from nanoscigpt.scientific.adapters import periodic_distances

    fractional = torch.tensor([[[0.95, 0.0, 0.0], [0.05, 0.0, 0.0]]])
    lattice = torch.eye(3).unsqueeze(0)
    distances = periodic_distances(fractional, lattice)

    assert torch.isclose(distances[0, 0, 1], torch.tensor(0.1), atol=1e-6)


def test_generate():
    cfg = GPTConfig(vocab_size=10, block_size=32, n_layer=1, n_head=1, n_embd=16)
    model = GPT(cfg)
    x = torch.randint(0, 10, (1, 4))
    out = model.generate(x, 8)
    assert out.shape == (1, 12)


def test_generate_stops_at_end_token_and_blocks_padding():
    cfg = GPTConfig(vocab_size=4, block_size=8, n_layer=1, n_head=1, n_embd=8)
    model = GPT(cfg)

    def forced_logits(self, idx, targets=None, pad_mask=None):
        logits = torch.zeros((idx.size(0), idx.size(1), 4))
        logits[:, -1, 1] = 20.0  # padding would win without the block
        logits[:, -1, 2] = 10.0  # end token should be selected next
        return logits, None

    model.forward = types.MethodType(forced_logits, model)
    output = model.generate(
        torch.tensor([[0]]),
        max_new_tokens=5,
        top_k=1,
        stop_token_id=2,
        forbidden_token_ids=[1],
    )

    assert output.tolist() == [[0, 2]]


def test_sampler_removes_end_token_from_display():
    from nanoscigpt.core.sampler import trim_at_token

    assert trim_at_token([3, 4, 2, 1], 2) == [3, 4]


def test_mask_tokens():
    from nanoscigpt.core.objectives import mask_tokens

    x = torch.tensor([[1, 2, 3, 4, 5]])
    x_m, y = mask_tokens(x, vocab_size=6, mask_prob=1.0)  # mask everything
    assert (y == x).all()  # all positions are targets
    assert (x_m == 6).all()  # all replaced by mask token id


def test_route_decision():
    from nanoscigpt.tasks.route_decision import QUESTIONS, decide

    r = decide({"data_scale": False, "task_sharing": True, "transfer_evidence": True, "multi_task_gain": True, "budget": True})
    assert r["route"] == "use_specialized_model"
    r2 = decide({"data_scale": True, "task_sharing": True, "transfer_evidence": True, "multi_task_gain": True, "budget": True})
    assert r2["route"] == "train_new_foundation"
    single_task = decide({"data_scale": True, "task_sharing": False, "transfer_evidence": True, "multi_task_gain": True, "budget": True})
    assert single_task["route"] == "use_specialized_model"
    assert single_task["failed_checks"][0] == "task_sharing"
    transfer_question = next(q for q in QUESTIONS if q["id"] == "transfer_evidence")
    assert "OOD留出" in transfer_question["question"]


def test_bidirectional_attention():
    from nanoscigpt.core.gpt import GPT, GPTConfig

    cfg = GPTConfig(vocab_size=10, block_size=16, n_layer=1, n_head=1, n_embd=16, causal=False)
    model = GPT(cfg)
    x = torch.randint(0, 10, (1, 8))
    # bidirectional: perturbing a later token changes earlier representations
    x2 = x.clone()
    x2[0, 4] = (x2[0, 4] + 1) % 10
    pos = torch.arange(8)
    h1 = model.transformer.wte(x) + model.transformer.wpe(pos)
    h2 = model.transformer.wte(x2) + model.transformer.wpe(pos)
    b1 = model.transformer.h[0](h1)
    b2 = model.transformer.h[0](h2)
    assert not torch.allclose(b1[0, 0], b2[0, 0])  # position 0 sees position 4


def test_objective_contrast_uses_dna_stream_data():
    from nanoscigpt.core.dataset import TokenStreamDataset
    from nanoscigpt.tasks.objective_contrast import get_training_dataset

    ds = get_training_dataset("dna", "data")
    assert isinstance(ds, TokenStreamDataset)


def test_objective_contrast_keeps_domain_outputs_separate():
    from pathlib import Path

    from nanoscigpt.tasks.objective_contrast import get_output_dir

    assert get_output_dir(None, "dna") == Path("out/objective_contrast/dna")
    assert get_output_dir("custom", "dna") == Path("custom")


def test_objective_contrast_is_reproducible_across_processes(tmp_path):
    outputs = []
    for run in ("a", "b"):
        out_dir = tmp_path / run
        subprocess.run(
            [
                sys.executable,
                "-m",
                "nanoscigpt.tasks.objective_contrast",
                "--domain",
                "dna",
                "--iters",
                "3",
                "--out_dir",
                str(out_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(json.loads((out_dir / "contrast_results.json").read_text()))
    assert outputs[0] == outputs[1]


def test_multihead_is_reproducible_across_processes(tmp_path):
    outputs = []
    for run in ("a", "b"):
        out_dir = tmp_path / run
        subprocess.run(
            [
                sys.executable,
                "-m",
                "nanoscigpt.tasks.multihead",
                "--epochs",
                "3",
                "--out_dir",
                str(out_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(json.loads((out_dir / "multihead_results.json").read_text()))
    assert outputs[0] == outputs[1]


def test_downstream_demo_summary_only_reports_that_the_flow_ran(tmp_path):
    from nanoscigpt.tasks.transfer_probe import downstream_demo_summary

    result_path = tmp_path / "probe_results.json"
    summary = downstream_demo_summary(0.967, result_path)

    assert summary == [
        "downstream task: completed",
        f"result saved: {result_path}",
    ]
    visible_text = "\n".join(summary).lower()
    assert "one-hot" not in visible_text
    assert "transfer" not in visible_text


def test_downstream_demo_result_only_keeps_the_classroom_run(tmp_path):
    from nanoscigpt.tasks.transfer_probe import downstream_demo_result

    result = downstream_demo_result(0.967)

    assert result == {"status": "completed", "downstream_score": 0.967}
