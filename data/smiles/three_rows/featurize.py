"""Compute structural features on an RDKit standardized parent molecule."""

import hashlib
import numpy as np
from rdkit.Chem import Descriptors as D, rdFingerprintGenerator

FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)


def features(m):
    count = np.zeros(96)
    for atom in m.GetAtoms():
        key = (
            atom.GetAtomicNum(),
            atom.GetIsAromatic(),
            atom.GetTotalNumHs(),
            atom.GetFormalCharge(),
            atom.GetDegree(),
            sum(n.GetAtomicNum() not in (1, 6) for n in atom.GetNeighbors()),
        )
        idx = int.from_bytes(
            hashlib.blake2b(repr(key).encode(), digest_size=4).digest(), "little"
        ) % len(count)
        count[idx] += 1
    basic = [
        0.0,
        m.GetNumHeavyAtoms(),
        D.MolWt(m) / 100,
        D.NumAromaticRings(m),
        D.NumHDonors(m),
        D.NumHAcceptors(m),
        D.NumRotatableBonds(m),
        D.TPSA(m) / 100,
        D.FractionCSP3(m),
    ]
    basic.extend(
        sum(a.GetAtomicNum() == e for a in m.GetAtoms())
        for e in (6, 7, 8, 9, 15, 16, 17, 35, 53)
    )
    return np.r_[basic, count, np.sqrt(FP.GetCountFingerprintAsNumPy(m))]
