"""Build the canonical (tanim x tarih) data contract panel and run its assertions.

Writes evidence to outputs/data_contract.json (this repo's convention: every reports/*.md
claim traces to a JSON file here). Does not touch the submission chain -- see
src/data/contract.py's module docstring for scope.

Usage: python scripts/build_data_contract.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.contract import (
    CALENDAR_VERSION,
    attach_origin_provenance,
    build_and_check,
    source_version,
)
from src.data.loader import DataLoader


def main() -> None:
    t0 = time.time()
    loader = DataLoader()
    train, test = loader.load_train_test(clean=True)  # declip() applied, per Section 2

    panel, findings = build_and_check(train, test)
    print(f"panel: {findings['n_rows']:,} rows, {findings['n_transformers']:,} transformers "
          f"[{time.time()-t0:.1f}s]")
    print(f"suspected_data_error rows: {findings['suspected_data_error_rows']}")
    print(f"test transformers with within-window date gaps: "
          f"{findings['test_transformers_with_within_window_gaps']} / "
          f"{findings['test_transformers_total']}")

    # Origin provenance at the real test cutoff, as a worked example / smoke test of
    # attach_origin_provenance -- this is the fold-aware half of the contract (Section 1's
    # "computable as of the fold cutoff" requirement).
    TEST_ORIGIN = "2026-04-01"
    with_prov = attach_origin_provenance(panel, TEST_ORIGIN)
    test_rows = with_prov.loc[with_prov["is_test"]]
    cold_start_share = float((~test_rows["transformer_seen_at_origin"]).mean())
    print(f"cold-start share of test rows at origin {TEST_ORIGIN}: {cold_start_share:.2%}")

    result = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "calendar_version": CALENDAR_VERSION,
        "source_version_train": source_version("data/train.csv"),
        "source_version_test": source_version("data/test.csv"),
        "findings": findings,
        "test_origin": TEST_ORIGIN,
        "cold_start_share_at_test_origin": cold_start_share,
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/data_contract.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"\nwrote outputs/data_contract.json [{time.time()-t0:.1f}s total]")


if __name__ == "__main__":
    main()
