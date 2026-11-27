"""Generate separate simulations for a warm-temperature scalar diffusion benchmark."""

import hashlib
from pathlib import Path
import numpy as np

SEEDS = (53, 71, 89, 107, 131)
BASE_TEMP = 298.0
GAS = 0.008314462618  # kJ/(mol K)


def simulate(n, seed, auxiliary=False):
    rng = np.random.default_rng(seed)
    porosity, crosslink = rng.uniform(0, 1, size=(2, n))
    temp = rng.uniform(308, 328, n)
    d0 = np.exp(rng.uniform(np.log(0.018), np.log(0.085), n))
    transition = 1 / (1 + np.exp(-(crosslink - 0.32 - 0.32 * porosity) / 0.065))
    energy = (
        18
        + 11 * crosslink
        + 9 * (1 - porosity)
        + 22 * transition
        + 6 * crosslink * (1 - porosity)
    )
    amplitude = rng.normal(0, 1, size=(n, 1, 3))
    times = np.arange(4)[None, :, None]
    modes = np.arange(1, 4)
    basis = np.sin(2 * np.pi * np.arange(64)[:, None] * modes / 64)
    fields = (amplitude * np.exp(-times * modes**2 * d0[:, None, None])) @ basis.T
    fields += rng.normal(0, 0.025, size=fields.shape)
    part = dict(
        obs=fields.astype("float32"),
        material=np.column_stack((porosity, crosslink)).astype("float32"),
        temp=temp.astype("float32"),
    )
    if auxiliary:
        # The 2D mean squared displacement divided by 4 estimates D at unit lag.
        # For 100 independent 2D increments its ratio to true D is chi2(200)/200.
        ref = d0 * rng.chisquare(200, n) / 200
        warm = (
            d0
            * np.exp(energy * (1 / BASE_TEMP - 1 / 318) / GAS)
            * rng.chisquare(200, n)
            / 200
        )
        part["msd"] = np.column_stack((4 * ref, 4 * warm)).astype("float32")
    else:
        part["y"] = (d0 * np.exp(energy * (1 / BASE_TEMP - 1 / temp) / GAS)).astype(
            "float32"
        )
    return part


def identities(part):
    return {hashlib.sha256(row.tobytes()).digest() for row in part["obs"]}


def prepare(folder):
    pool = simulate(1536, 693101)
    for s in SEEDS:
        ids = np.random.default_rng(s).choice(len(pool["y"]), 32, replace=False)
        np.savez_compressed(folder / f"{s}.npz", **{k: a[ids] for k, a in pool.items()})
    aux, dev = simulate(2400, 693121, True), simulate(384, 693109)
    seen, aid, did = map(identities, (pool, aux, dev))
    assert len(seen) == 1536 and len(aid) == 2400 and len(did) == 384
    assert not (seen & aid or seen & did or aid & did)
    np.savez_compressed(folder / "aux.npz", **aux)
    np.savez_compressed(folder / "dev.npz", **dev)
