"""Print the source-row mapping for every generated batch.

This is a small operator aid: it shows the batch number, source row, canonical
complaint question and the intent that ``clean_data.py`` will use.

Usage::

    python diagnostic.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_pipeline"))

from source_metadata import build_batch_metadata  # noqa: E402


def main() -> None:
    batches = build_batch_metadata(".")
    print(f"{'batch':>5}  {'row':>4}  {'intent':<32}  topic / question")
    for item in batches:
        print(
            f"{item.batch_number:>5}  {str(item.source_row):>4}  "
            f"{str(item.intent_code):<32}  "
            f"{item.topic}: {item.question!r}"
        )
        for flag in item.flags:
            print(f"       ! {flag}")


if __name__ == "__main__":
    main()
