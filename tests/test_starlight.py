"""Tests use synthetic fixtures only; real benchmark scores are separate."""
import numpy as np
import pytest
import torch


def test_flat_head_preserves_order_and_pretraining_causality():
    from nanoscigpt.starlight import StarLightGPT
    model = StarLightGPT(patch_size=4, width=16, layers=1, heads=2, max_length=16, head_mode='flat').eval()
    x = torch.linspace(-1, 1, 16).reshape(1, 16)
    assert model(x).shape == (1, 3)
    assert not torch.allclose(model(x), model(x.flip(-1)))
    altered = x.clone()
    altered[:, 8:] = 99
    torch.testing.assert_close(model.predict_patches(x)[:, :2], model.predict_patches(altered)[:, :2])
    assert model.config['head_mode'] == 'flat'


def test_validation_only_does_not_require_or_evaluate_test_data(tmp_path):
    from nanoscigpt.starlight import run, Config
    # No test arrays at all: accessing them during tuning must fail this test.
    source = tmp_path / 'train-only.npz'
    np.savez(source, x_train=np.random.default_rng(1).normal(size=(18, 16)).astype('float32'),
             y_train=np.tile([1, 2, 3], 6), fit_indices=np.arange(12), validation_indices=np.arange(12, 18))
    out = tmp_path / 'validation'
    result = run(Config(str(source), str(out), pretrain_steps=2, finetune_steps=2,
                        batch_size=3, patch_size=4, width=16, layers=1, eval_every=1,
                        validation_only=True, head_mode='flat', cosine=True))
    assert result['metrics'] == {}
    assert set(result['validation']) == {'scratch', 'scratch_long', 'finetuned'}
    assert not (out / 'predictions.npz').exists()
    history = __import__('json').loads((out / 'history.json').read_text())
    fine = [h for h in history if h['stage'] == 'finetuned']
    assert fine[-1]['learning_rate'] < fine[0]['learning_rate']
    scratch = [h for h in history if h['stage'] == 'scratch']
    assert [h['learning_rate'] for h in scratch] == [h['learning_rate'] for h in fine]


def test_single_curve_three_classes_and_causal_prediction():
    from nanoscigpt.starlight import StarLightGPT
    model = StarLightGPT(patch_size=4, width=16, layers=1, heads=2, max_length=16).eval()
    x = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 32
    assert model(x).shape == (2, 3)
    changed = x.clone()
    changed[:, 8:] = 99
    torch.testing.assert_close(model.predict_patches(x)[:, :2], model.predict_patches(changed)[:, :2])
    loss = model.pretraining_loss(x) + torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 2]))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_original_split_and_labels_are_preserved(tmp_path):
    from nanoscigpt.starlight import load_data
    source = tmp_path / 'data.npz'
    np.savez(source, x_train=np.arange(24).reshape(6, 4).astype('float32'),
             y_train=[1, 2, 3, 1, 2, 3], x_test=np.ones((3, 4), dtype='float32'),
             y_test=[3, 1, 2], fit_indices=[0, 1, 2], validation_indices=[3, 4, 5])
    data = load_data(source)
    assert data['fit'][1].tolist() == [0, 1, 2]
    assert data['test'][1].tolist() == [2, 0, 1]
    assert data['val'][0][0].tolist() == [12, 13, 14, 15]


def test_short_run_writes_reloadable_models_and_rejects_overwrite(tmp_path):
    from nanoscigpt.starlight import run, Config, StarLightGPT
    source = tmp_path / 'fixture.npz'
    rng = np.random.default_rng(3)
    np.savez(source, x_train=rng.normal(size=(18, 16)).astype('float32'),
             y_train=np.tile([1, 2, 3], 6), x_test=rng.normal(size=(6, 16)).astype('float32'),
             y_test=[1, 2, 3, 1, 2, 3], fit_indices=np.arange(12), validation_indices=np.arange(12, 18))
    out = tmp_path / 'run'
    config = Config(str(source), str(out), pretrain_steps=2, finetune_steps=2,
                    eval_every=1, batch_size=3, patch_size=4, width=16, layers=1)
    result = run(config)
    assert result['budget']['scratch_long'] == 4
    assert result['budget']['scratch'] == result['budget']['finetuned'] == 2
    pred = np.load(out / 'predictions.npz')
    assert pred['y_true'].tolist() == [1, 2, 3, 1, 2, 3]
    for branch in ('scratch', 'scratch_long', 'finetuned'):
        ckpt = torch.load(out / (branch + '.pt'), weights_only=True)
        model = StarLightGPT(**ckpt['config']).eval()
        model.load_state_dict(ckpt['state'])
        with torch.no_grad():
            p = model(torch.from_numpy(np.load(source)['x_test'])).softmax(-1).numpy()
        np.testing.assert_allclose(p, pred[branch], atol=1e-6)
        assert len(result['metrics'][branch]['confusion_matrix']) == 3
    with pytest.raises(FileExistsError):
        run(config)
