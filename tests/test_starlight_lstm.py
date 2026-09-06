"""LSTM must train on the same split and reload its actual predictions."""
import numpy as np
import pytest
import torch


def test_lstm_reads_every_patch_and_backpropagates():
    from nanoscigpt.starlight_lstm import LightCurveLSTM
    model = LightCurveLSTM(patch_size=4, width=8, layers=1)
    x = torch.randn(3, 16, requires_grad=True)
    logits = model(x)
    assert logits.shape == (3, 3)
    logits.sum().backward()
    assert x.grad[:, :4].abs().sum() > 0
    assert x.grad[:, -4:].abs().sum() > 0
    with pytest.raises(ValueError, match='divisible'):
        model(torch.randn(3, 17))


def test_training_reloads_and_test_labels_do_not_select_checkpoint(tmp_path):
    from nanoscigpt.starlight_lstm import Config, LightCurveLSTM, run
    from nanoscigpt.starlight import probabilities
    rng = np.random.default_rng(9)
    x, xt = rng.normal(size=(18, 16)).astype('float32'), rng.normal(size=(6, 16)).astype('float32')
    y, yt = np.tile([1, 2, 3], 6), np.tile([1, 2, 3], 2)
    outputs = []
    for name, labels in [('a', yt), ('b', np.roll(yt, 1))]:
        source, out = tmp_path / f'{name}.npz', tmp_path / name
        np.savez(source, x_train=x, y_train=y, fit_indices=np.arange(12),
                 validation_indices=np.arange(12, 18), x_test=xt, y_test=labels)
        cfg = Config(str(source), str(out), steps=4, eval_every=2, batch_size=6,
                     patch_size=4, width=8, layers=1, threads=1)
        report = run(cfg)
        checkpoint = torch.load(out / 'lstm.pt', weights_only=True)
        model = LightCurveLSTM(**checkpoint['config'])
        model.load_state_dict(checkpoint['state'])
        with np.load(out / 'predictions.npz') as saved:
            np.testing.assert_array_equal(saved['y_true'], labels)
            np.testing.assert_allclose(saved['lstm'], probabilities(model, torch.from_numpy(xt), 6))
        assert report['fit_samples'] == 12
        assert report['selected_step'] in (2, 4)
        outputs.append(checkpoint)
        with pytest.raises(FileExistsError):
            run(cfg)
    assert outputs[0]['step'] == outputs[1]['step']
    for key in outputs[0]['state']:
        torch.testing.assert_close(outputs[0]['state'][key], outputs[1]['state'][key], rtol=0, atol=0)
