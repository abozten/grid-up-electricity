"""Section 1 ("Freeze the data contract") + Section 2 ("Clean target history carefully") of
the ASHRAE-derived preprocessing checklist, adapted to this repo's actual data:

    data/train.csv  -- 2025-01-01..2026-03-31, one row per (tanim, tarih) observed reading
    data/test.csv   -- 2026-04-01..2026-07-31 (horizon 1..122 days), one row per scored key

This is a diagnostics / QA layer. It does NOT feed `src/final_v33.py` or any shipped
submission chain -- see the README warning about the two parallel code paths. Its job is to
give every other script one canonical, provenance-stamped, assertion-checked table to build
features from, and to make target-cleaning decisions auditable rather than ad hoc.

Deliberately NOT built here (separate follow-on work, needs the walk-forward fold machinery
in `src/_archive/legacy/pipeline4.py` / `src/evaluation/oof.py` wired in):
    - Section 3: weather/timestamp alignment-shift search, EPİAŞ availability classing
    - Section 4: leakage-safe leave-one-out cohort aggregates, shrinkage
    - Section 5: per-fold train/validation/test table splits, feature manifest, population-
      shift and out-of-support reporting

Column vocabulary of the raw data (Turkish, kept as-is rather than translated, since every
other module in this repo uses these names):
    tanim   transformer id            guc      nameplate capacity (kVA)
    tarih   date                      tuketim  daily consumption (kWh) -- the target
    lokasyon  '>'-delimited location hierarchy (il > [bolge >] ilce)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import DataLoader

ID_COL = "tanim"
DATE_COL = "tarih"
TARGET_COL = "tuketim"

CALENDAR_VERSION = "cal-v1"  # bump if holiday/calendar-derivation logic changes
LONG_ZERO_RUN_DAYS = 28  # threshold for the `long_zero_run` flag


# ---------------------------------------------------------------------------
# Provenance stamps
# ---------------------------------------------------------------------------

def source_version(path: str | Path) -> str:
    """Cheap content-fingerprint for a raw source file: sha1 of (size, mtime_ns).

    Not a full hash of file contents (these files can be large) -- good enough to detect
    "this feature table was built from a different data pull than that one," which is the
    only thing `source_version_weather` / `source_version_epias` are for.
    """
    p = Path(path)
    if not p.exists():
        return "missing"
    st = p.stat()
    return hashlib.sha1(f"{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Section 1: canonical panel
# ---------------------------------------------------------------------------

def find_raw_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Ingestion-duplication check on a RAW (pre-contract) frame, before any groupby collapses
    them silently. Returns the duplicate rows (both copies) for inspection."""
    dup_mask = df.duplicated(subset=[ID_COL, DATE_COL], keep=False)
    return df.loc[dup_mask].sort_values([ID_COL, DATE_COL])


def build_canonical_panel(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """One row per (tanim, tarih) across train UNION test, with immutable provenance fields.

    `train` and `test` are expected already location-parsed (il/bolge/ilce present) -- pass
    the output of `DataLoader.load_train_test()`. Declipping is a target-history decision
    (Section 2) and must have already happened on `train` before calling this, so this
    function only ever describes the target it was handed, never edits it.
    """
    dup_tr = find_raw_duplicate_rows(train)
    dup_te = find_raw_duplicate_rows(test)
    if len(dup_tr) or len(dup_te):
        raise AssertionError(
            f"duplicate transformer_id x date rows before contract build: "
            f"{len(dup_tr)} in train, {len(dup_te)} in test"
        )

    tr = train.copy()
    tr["is_train"] = True
    tr["is_test"] = False

    te = test.copy()
    te["is_train"] = False
    te["is_test"] = True
    if TARGET_COL not in te.columns:
        te[TARGET_COL] = np.nan

    keep = [ID_COL, DATE_COL, "guc", "il", "bolge", "ilce", TARGET_COL, "is_train", "is_test"]
    if "id" in te.columns:
        keep_te = keep + ["id"]
    else:
        keep_te = keep

    panel = pd.concat([tr[keep], te[keep_te]], ignore_index=True, sort=False)
    panel = panel.sort_values([ID_COL, DATE_COL], kind="mergesort").reset_index(drop=True)

    panel["log_t"] = np.log1p(panel[TARGET_COL])
    panel["calendar_version"] = CALENDAR_VERSION
    return panel


# ---------------------------------------------------------------------------
# Section 2: target-history flags (preserve zeros, flag only defensible errors)
# ---------------------------------------------------------------------------

def add_target_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """Row-level target/availability flags. Purely descriptive of the value the row already
    has -- never imputes, never deletes, never converts a zero to missing."""
    panel = panel.copy()
    observed = panel["is_train"] & panel[TARGET_COL].notna()
    panel["target_observed"] = observed
    panel["target_missing"] = panel["is_train"] & ~observed
    panel["target_true_zero"] = observed & (panel[TARGET_COL] == 0)

    is_finite = np.isfinite(panel[TARGET_COL].fillna(0.0))
    panel["suspected_data_error"] = observed & (
        (panel[TARGET_COL] < 0) | ~is_finite
    )
    return panel


def history_asof(panel: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Per-transformer target-history summary using strictly rows with tarih < origin.

    This is the fold-aware half of the contract: call it once per walk-forward cutoff (each
    entry of `PARAMS.train_blocks` / `FOLDS`, plus the real test origin 2026-04-01) rather than
    baking a single global answer into the panel. Returns one row per `tanim` that appeared
    at all before `origin`; transformers with none are cold-start and simply absent here --
    callers should treat a missing join as `transformer_seen_at_origin=False`.
    """
    origin = pd.Timestamp(origin)
    hist = panel.loc[
        panel["target_observed"] & (panel[DATE_COL] < origin), [ID_COL, DATE_COL, "target_true_zero"]
    ].sort_values([ID_COL, DATE_COL])
    if hist.empty:
        return pd.DataFrame(columns=[
            ID_COL, "first_observed_date", "last_observed_date", "n_history_days_at_origin",
            "ever_positive_before_origin", "days_since_last_positive", "zero_share_history",
            "all_observed_history_zero", "history_contains_only_zero_or_missing", "long_zero_run",
        ])

    g = hist.groupby(ID_COL, sort=False)
    out = g.agg(
        first_observed_date=(DATE_COL, "min"),
        last_observed_date=(DATE_COL, "max"),
        n_history_days_at_origin=(DATE_COL, "count"),
        zero_share_history=("target_true_zero", "mean"),
    ).reset_index()

    pos = hist.loc[~hist["target_true_zero"]]
    last_positive = pos.groupby(ID_COL, sort=False)[DATE_COL].max()
    out["ever_positive_before_origin"] = out[ID_COL].map(last_positive.notna().reindex(
        out[ID_COL], fill_value=False)).astype(bool)
    out["days_since_last_positive"] = out[ID_COL].map(
        (origin - last_positive).dt.days
    )  # NaN if never positive -- left as NaN, not inf/sentinel, per "don't invent a value" spirit

    out["all_observed_history_zero"] = out["zero_share_history"] >= 1.0
    # This dataset has no independently-flagged "missing" reporting gaps (see
    # `assert_test_window_contiguity` for the closest available signal), so the "or_missing"
    # variant currently coincides with the plain zero-history flag; keep both names so callers
    # written against the checklist's vocabulary don't need to change if gap-detection is
    # added later.
    out["history_contains_only_zero_or_missing"] = out["all_observed_history_zero"]

    def _max_zero_run(s: pd.Series) -> int:
        z = s.to_numpy()
        best = cur = 0
        for v in z:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    runs = g["target_true_zero"].apply(_max_zero_run)
    out["long_zero_run"] = out[ID_COL].map(runs) >= LONG_ZERO_RUN_DAYS

    return out


def attach_origin_provenance(panel: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Merge `history_asof(panel, origin)` onto every row of `panel`, plus the row-level
    provenance fields (`forecast_origin`, `horizon`, `transformer_seen_at_origin`).

    Safe to call with `origin` equal to a training fold's validation cutoff OR the real test
    origin (2026-04-01) -- the causal cutoff is the only thing that changes.
    """
    origin = pd.Timestamp(origin)
    hist = history_asof(panel, origin)
    out = panel.merge(hist, on=ID_COL, how="left")
    out["forecast_origin"] = origin
    out["horizon"] = (out[DATE_COL] - origin).dt.days
    out["transformer_seen_at_origin"] = out[ID_COL].isin(hist[ID_COL])
    out["n_history_days_at_origin"] = out["n_history_days_at_origin"].fillna(0).astype(int)
    return out


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_contract(panel: pd.DataFrame) -> dict:
    """Run the checklist's Section-1 assertions. Raises AssertionError on any unambiguous
    invariant violation; returns a dict of soft findings (counts, not necessarily errors) for
    the ones the checklist says to inspect rather than hard-fail on (e.g. within-window date
    gaps, which are a real property of this panel -- see `PREPROCESSING_CHECKLIST` Section 2's
    "separate missingness from zero").
    """
    findings: dict = {}

    dup = panel.duplicated(subset=[ID_COL, DATE_COL]).sum()
    if dup:
        raise AssertionError(f"{dup} duplicate transformer_id x date rows in canonical panel")

    neg = (panel.loc[panel["target_observed"], TARGET_COL] < 0).sum()
    if neg:
        raise AssertionError(f"{neg} negative kWh rows in observed target")

    bad_guc = (panel["guc"] <= 0) | panel["guc"].isna()
    if bad_guc.any():
        raise AssertionError(f"{int(bad_guc.sum())} rows with unrecognized/invalid guc")

    bad_loc = panel[["il", "bolge", "ilce"]].isna().any(axis=1)
    if bad_loc.any():
        raise AssertionError(f"{int(bad_loc.sum())} rows with unrecognized region hierarchy")

    findings["n_rows"] = len(panel)
    findings["n_transformers"] = panel[ID_COL].nunique()
    findings["suspected_data_error_rows"] = int(panel["suspected_data_error"].sum())

    # Soft check: within each transformer's OWN test-window [min, max] scored date range,
    # are all calendar days present? Gaps here are real (successor/replacement transformers,
    # mid-horizon commissioning) -- flag, don't fail.
    te = panel.loc[panel["is_test"]]
    def _contiguous(dates: pd.Series) -> bool:
        d = dates.sort_values()
        full = pd.date_range(d.min(), d.max(), freq="D")
        return len(full) == len(d)
    gap_check = te.groupby(ID_COL)[DATE_COL].apply(_contiguous)
    findings["test_transformers_with_within_window_gaps"] = int((~gap_check).sum())
    findings["test_transformers_total"] = int(len(gap_check))

    return findings


def build_and_check(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convenience entry point: build the panel, add target flags, run assertions."""
    panel = build_canonical_panel(train, test)
    panel = add_target_flags(panel)
    findings = assert_contract(panel)
    return panel, findings
