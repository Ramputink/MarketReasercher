"""
forward_test.preregister — write the immutable pre-registration manifest.

Run ONCE to stamp the frozen configs + hash. Re-running refuses to overwrite an
existing manifest unless --force is passed (the whole point is immutability).

Run:  python3 -m forward_test.preregister
"""
from __future__ import annotations

import json
import sys

from forward_test import (PREREG_PATH, REGISTERED_DATE, FORWARD_START, EXCHANGE,
                          PREREGISTERED, config_hash)


def main():
    import os
    if os.path.exists(PREREG_PATH) and "--force" not in sys.argv:
        existing = json.load(open(PREREG_PATH))
        print(f"pre-registration already exists (hash {existing['hash'][:12]}…, "
              f"registered {existing['registered_date']}).")
        print("It is meant to be immutable. Use --force only to re-stamp deliberately.")
        return existing
    manifest = {
        "registered_date": REGISTERED_DATE,
        "forward_start": FORWARD_START,
        "exchange": EXCHANGE,
        "configs": PREREGISTERED,
        "hash": config_hash(),
        "protocol": ("Score each config ONLY on data strictly after forward_start. "
                     "Never re-fit. Append every run to forward_log.jsonl. The "
                     "forward Sharpe/return accumulate as a true out-of-sample "
                     "track record."),
    }
    json.dump(manifest, open(PREREG_PATH, "w"), indent=2)
    print(f"wrote {PREREG_PATH}")
    print(f"hash: {manifest['hash']}")
    print(f"forward window opens after {FORWARD_START}")
    return manifest


if __name__ == "__main__":
    main()
