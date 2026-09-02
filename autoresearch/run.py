"""Backward-compatible entry point for the current three-stage pipeline.

Use ``python -m autoresearch.pipeline`` in new classroom material.  Keeping
this alias means older notes that call ``autoresearch.run`` still execute the
same maintained flow instead of the retired protein-only probe path.
"""

from .pipeline import main


if __name__ == "__main__":
    main()
