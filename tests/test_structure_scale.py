import numpy as np

from nanoscigpt.tasks.structure_scale import chain_panel


def test_end_to_end_distance_is_absent_from_both_symmetric_input_blocks():
    visible, target = chain_panel(32, [264130, 1])
    assert visible.shape == (32, 14, 16)
    assert np.all(target > 0)
    assert not (visible.numpy() == target[:, None, None]).any()
