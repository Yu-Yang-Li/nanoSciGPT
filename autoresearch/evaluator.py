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
}


def evaluate_loss_gain(baseline, candidate, threshold=None):
    """Compare two loss values without rounding before the decision."""
    threshold = CRITERIA["train_improve_min"] if threshold is None else float(threshold)
    baseline = float(baseline)
    candidate = float(candidate)
    delta = baseline - candidate
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "threshold": threshold,
        "direction": "lower_is_better",
        "passed": delta >= threshold,
    }


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def evaluate_train(domain):
    """Did training actually run, and did it pass the loss criterion?"""
    structured = domain in {"weather", "crystal", "structure3d", "image", "spectrum", "field"}
    base = REPO_ROOT / "out" / domain
    log_path = base / "model" / "train_log.json" if structured else base / "train_log.json"
    ckpt_path = base / "model" / "ckpt.pt" if structured else base / "ckpt.pt"
    if not ckpt_path.exists():
        return {"level": "ran", "passed": False, "reason": "checkpoint missing"}
    log = _read_json(log_path)
    metric_key = "pretrain_val_loss" if structured else "best_val_loss"
    if not log or metric_key not in log:
        return {"level": "ran", "passed": False, "reason": "train_log.json incomplete"}
    loss = log[metric_key]
    passed = loss < CRITERIA["train_val_loss_max"]
    metric = "pretrain_loss" if structured else "best_val_loss"
    return {"level": "evaluated", "passed": passed, "metric": metric,
            "value": round(loss, 4),
            "reason": f"{metric.replace('_', ' ')} {loss:.4f} {'<' if passed else '>='} {CRITERIA['train_val_loss_max']}"}


def evaluate_train_gain(domain, baseline_loss):
    """Did extending the budget improve on the V0 baseline by enough?"""
    structured = domain in {"weather", "crystal", "structure3d", "image", "spectrum", "field"}
    log_path = REPO_ROOT / "out" / domain
    log_path = log_path / "model" / "train_log.json" if structured else log_path / "train_log.json"
    log = _read_json(log_path)
    metric_key = "pretrain_val_loss" if structured else "best_val_loss"
    if not log or metric_key not in log:
        return {"level": "ran", "passed": False, "reason": "no train_log.json after extended run"}
    delta = baseline_loss - log[metric_key]
    passed = delta >= CRITERIA["train_improve_min"]
    return {"level": "evaluated", "passed": passed, "metric": "loss_gain_vs_v0",
            "value": round(delta, 4), "baseline": round(baseline_loss, 4),
            "reason": f"gain {delta:+.4f} vs required {CRITERIA['train_improve_min']}"}
