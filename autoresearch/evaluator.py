"""Formal evaluator: the only place that decides whether a run produced evidence.

Evidence levels (the class boundary the course insists on):

- design             : a plan exists, nothing was executed
- ran                : a tool completed and declared outputs exist
- evaluated          : outputs were checked against numeric acceptance criteria
- externally_verified: independent/peer-reviewed evidence (never claimed by this repo)

The evaluator returns structured results; it never invents numbers.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# acceptance thresholds - deliberately conservative teaching values
CRITERIA = {
    "train_val_loss_max": 6.0,       # any loss below this counts as "learned something"
    "train_improve_min": 0.05,       # extended training must beat V0 by at least this
    "transfer_gain_min": 0.05,       # probe must beat one-hot by at least this
}


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def evaluate_train(domain):
    """Did training actually run, and did it pass the loss criterion?"""
    log_path = REPO_ROOT / "out" / domain / "train_log.json"
    ckpt_path = REPO_ROOT / "out" / domain / "ckpt.pt"
    if not ckpt_path.exists():
        return {"level": "ran", "passed": False, "reason": "checkpoint missing"}
    log = _read_json(log_path)
    if not log or "best_val_loss" not in log:
        return {"level": "ran", "passed": False, "reason": "train_log.json incomplete"}
    loss = log["best_val_loss"]
    passed = loss < CRITERIA["train_val_loss_max"]
    return {"level": "evaluated", "passed": passed, "metric": "best_val_loss",
            "value": round(loss, 4),
            "reason": f"val loss {loss:.4f} {'<' if passed else '>='} {CRITERIA['train_val_loss_max']}"}


def evaluate_train_gain(domain, baseline_loss):
    """Did extending the budget improve on the V0 baseline by enough?"""
    log = _read_json(REPO_ROOT / "out" / domain / "train_log.json")
    if not log or "best_val_loss" not in log:
        return {"level": "ran", "passed": False, "reason": "no train_log.json after extended run"}
    delta = baseline_loss - log["best_val_loss"]
    passed = delta >= CRITERIA["train_improve_min"]
    return {"level": "evaluated", "passed": passed, "metric": "loss_gain_vs_v0",
            "value": round(delta, 4), "baseline": round(baseline_loss, 4),
            "reason": f"gain {delta:+.4f} vs required {CRITERIA['train_improve_min']}"}


def evaluate_transfer_probe():
    """Does our pretrained encoder beat one-hot on the probe task?"""
    res = _read_json(REPO_ROOT / "out" / "transfer_probe" / "probe_results.json")
    if not res:
        return {"level": "ran", "passed": False, "reason": "probe_results.json missing"}
    delta = res.get("pretrained_encoder", 0.0) - res.get("onehot", 1.0)
    passed = delta >= CRITERIA["transfer_gain_min"]
    return {"level": "evaluated", "passed": passed, "metric": "transfer_delta",
            "value": round(delta, 3), "detail": res,
            "reason": (f"transfer delta {delta:+.3f}; on a 450-sequence fixture a negative/"
                       f"small delta is the EXPECTED honest result") if not passed else
                      f"transfer delta {delta:+.3f} passes"}
