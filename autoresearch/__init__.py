"""autoresearch: a minimal virtual-environment AI Scientist that operates nanoSciGPT.

It only acts through declared tool contracts, evaluates every result with a
formal evaluator, and persists research state across rounds. Teaching map:

- tools.py     -> tool contract (executable step + declared I/O + budget)
- evaluator.py -> formal evaluator (ran vs. valid is defined here)
- state.py     -> cross-round research state (hypothesis, evidence, questions)
- run.py       -> the loop: feedback changes the next step; human gate included
"""
