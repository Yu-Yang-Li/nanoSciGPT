"""Replay the archived DSK proposals for real GD-1 by integrating each orbit on CPU."""
import csv
from pathlib import Path

import numpy as np

from .stream_fit import NAMES, chi2


def main():
    path = Path(__file__).resolve().parents[1] / "docs/stream-skill/real-candidates.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        proposals = list(csv.DictReader(handle))
    print("skill,batch,evaluations,best_total_chi2_so_far")
    for skill in ("empty", "final"):
        best = np.inf
        count = 0
        for batch in range(1, 5):
            rows = [r for r in proposals if r["skill"] == skill and int(r["batch"]) == batch]
            theta = [[float(r[name]) for name in NAMES] for r in rows]
            best = min(best, chi2(theta).sum(axis=1).min())
            count += len(rows)
            print(f"{skill},{batch},{count},{best:.2f}")


if __name__ == "__main__":
    main()
