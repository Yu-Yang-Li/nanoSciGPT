"""Cross-round research state: what the scientist remembers between rounds.

Each round appends evidence with an explicit level (design/ran/evaluated/
externally_verified). Hypotheses carry status: proposed -> supported /
refuted / inconclusive. Open questions drive the next round.

This file is how "research state survives across rounds" is made concrete:
a JSON file, not vibes.
"""

import json
from pathlib import Path


class ResearchState:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"rounds": [], "hypotheses": [], "evidence": [],
                         "open_questions": [], "next_action": None,
                         "conclusions": [], "boundary_notes": []}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def add_round(self, rnd, action, note):
        self.data["rounds"].append({"round": rnd, "action": action, "note": note})

    def add_hypothesis(self, hid, text):
        if any(h["id"] == hid for h in self.data["hypotheses"]):
            return
        self.data["hypotheses"].append({"id": hid, "text": text, "status": "proposed"})

    def set_hypothesis(self, hid, status, why):
        for h in self.data["hypotheses"]:
            if h["id"] == hid:
                h["status"] = status
                h["resolution"] = why

    def add_evidence(self, rnd, tool, level, passed, summary):
        self.data["evidence"].append({"round": rnd, "tool": tool, "level": level,
                                      "passed": bool(passed), "summary": summary})

    def add_question(self, q):
        if q not in self.data["open_questions"]:
            self.data["open_questions"].append(q)

    def close_question(self, q):
        if q in self.data["open_questions"]:
            self.data["open_questions"].remove(q)

    def set_next_action(self, action):
        self.data["next_action"] = action

    def add_conclusion(self, text):
        self.data["conclusions"].append(text)

    def add_boundary_note(self, text):
        self.data["boundary_notes"].append(text)
