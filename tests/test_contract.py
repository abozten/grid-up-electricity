"""Tests for src/data/contract.py -- the Section 1/2 data-contract diagnostics layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.contract import (
    add_target_flags,
    assert_contract,
    attach_origin_provenance,
    build_canonical_panel,
    find_raw_duplicate_rows,
    history_asof,
)


def _toy_train():
    rows = []
    # t1: positive history, then a long zero run
    for i, day in enumerate(pd.date_range("2025-01-01", periods=40)):
        v = 10.0 if i < 10 else 0.0
        rows.append(dict(tanim="t1", guc=100, tarih=day, tuketim=v, il="A", bolge="A", ilce="A1"))
    # t2: always positive
    for day in pd.date_range("2025-01-01", periods=5):
        rows.append(dict(tanim="t2", guc=200, tarih=day, tuketim=5.0, il="A", bolge="A", ilce="A1"))
    return pd.DataFrame(rows)


def _toy_test():
    rows = []
    for day in pd.date_range("2025-02-15", periods=3):
        rows.append(dict(tanim="t1", guc=100, tarih=day, il="A", bolge="A", ilce="A1"))
    # t3 is cold-start: never in train
    rows.append(dict(tanim="t3", guc=50, tarih=pd.Timestamp("2025-02-15"), il="A", bolge="A", ilce="A1"))
    return pd.DataFrame(rows)


def test_build_canonical_panel_no_duplicates():
    panel = build_canonical_panel(_toy_train(), _toy_test())
    assert not panel.duplicated(subset=["tanim", "tarih"]).any()
    assert panel["is_train"].sum() == 45
    assert panel["is_test"].sum() == 4


def test_find_raw_duplicate_rows_detects_ingestion_dupes():
    tr = _toy_train()
    dup = pd.concat([tr, tr.iloc[[0]]], ignore_index=True)
    found = find_raw_duplicate_rows(dup)
    assert len(found) == 2


def test_target_flags_preserve_zeros():
    panel = build_canonical_panel(_toy_train(), _toy_test())
    panel = add_target_flags(panel)
    t1 = panel[(panel.tanim == "t1") & panel.is_train].sort_values("tarih")
    # the trailing zeros must stay `target_true_zero`, never become NaN/missing
    assert t1.iloc[-1]["target_true_zero"]
    assert t1.iloc[-1]["target_observed"]
    assert not t1.iloc[-1]["target_missing"]


def test_history_asof_is_causal_and_flags_long_zero_run():
    panel = build_canonical_panel(_toy_train(), _toy_test())
    panel = add_target_flags(panel)
    origin = pd.Timestamp("2025-02-15")
    hist = history_asof(panel, origin).set_index("tanim")

    assert hist.loc["t1", "n_history_days_at_origin"] == 40
    assert hist.loc["t1", "long_zero_run"]
    assert hist.loc["t1", "ever_positive_before_origin"]
    assert not hist.loc["t2", "long_zero_run"]
    assert "t3" not in hist.index  # cold-start: no history before origin

    # causality: nothing at/after origin may influence the summary
    earlier = history_asof(panel, pd.Timestamp("2025-01-05")).set_index("tanim")
    assert earlier.loc["t1", "n_history_days_at_origin"] == 4


def test_attach_origin_provenance_marks_cold_start():
    panel = build_canonical_panel(_toy_train(), _toy_test())
    panel = add_target_flags(panel)
    out = attach_origin_provenance(panel, "2025-02-15")
    t3_row = out[(out.tanim == "t3") & out.is_test].iloc[0]
    assert not t3_row["transformer_seen_at_origin"]
    assert t3_row["n_history_days_at_origin"] == 0
    t1_row = out[(out.tanim == "t1") & out.is_test].iloc[0]
    assert t1_row["transformer_seen_at_origin"]
    assert t1_row["horizon"] == 0


def test_assert_contract_rejects_negative_kwh():
    tr = _toy_train()
    tr.loc[0, "tuketim"] = -5.0
    panel = build_canonical_panel(tr, _toy_test())
    panel = add_target_flags(panel)
    with pytest.raises(AssertionError, match="negative kWh"):
        assert_contract(panel)


def test_assert_contract_passes_on_clean_toy_data():
    panel = build_canonical_panel(_toy_train(), _toy_test())
    panel = add_target_flags(panel)
    findings = assert_contract(panel)
    assert findings["n_transformers"] == 3
    assert findings["suspected_data_error_rows"] == 0
