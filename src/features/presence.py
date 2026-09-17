"""Presence-pattern features: which days of the scored window a transformer reports on.

This is information the test file hands over before any prediction is made -- test.csv
states exactly which (tanim, tarih) pairs must be predicted, so the shape of a
transformer's row calendar inside the scored window is legitimately observable. It is not
history and it is not leakage: no consumption value enters, only the existence of rows.

It was found in the cold-start level study (reports/ORACLE_LEVEL.md): out of five probes
for per-transformer information beyond capacity x district, this was the only one that
paid. On a transformer holdout it lifted level R2 from 0.164 to 0.437, because a
transformer that stops reporting mid-window is dying and its consumption collapses
(RMSLE of the capacity prior on that cohort: 5.53).

`ends_early` and `starts_late` are kept as separate columns rather than folded into a
single coverage fraction, because they mean opposite things: a transformer that stops
early is dying (mean level 0.87 in log space), one that starts late is newly energised
and perfectly normal (6.05).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EDGE_DAYS = 14          # how far from a window edge counts as "early" / "late"

PRESENCE_COLS = ["pres_frac", "pres_start_lag", "pres_end_lead", "pres_ends_early",
                 "pres_starts_late", "pres_gap_frac"]


class PresenceFeatureExtractor:
    """Row-calendar shape of each transformer inside the block being predicted."""

    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        """Attach presence columns, measured over the window `df` itself spans.

        Applied to a training snapshot block the window is that block's date range; applied
        to the test block it is the test date range. Both are known without looking at any
        target value.
        """
        df = df.copy()
        start, end = df["tarih"].min(), df["tarih"].max()
        span = max((end - start).days + 1, 1)

        g = df.groupby("tanim", observed=True)["tarih"]
        n_days = g.transform("size").astype(np.float64)
        first = g.transform("min")
        last = g.transform("max")

        df["pres_frac"] = n_days / span
        df["pres_start_lag"] = (first - start).dt.days / span
        df["pres_end_lead"] = (end - last).dt.days / span
        df["pres_ends_early"] = ((end - last).dt.days > EDGE_DAYS).astype(np.int8)
        df["pres_starts_late"] = ((first - start).dt.days > EDGE_DAYS).astype(np.int8)

        # Days missing from the interior of the transformer's own live span: an outage
        # that it came back from, as opposed to an edge truncation.
        live_span = ((last - first).dt.days + 1).clip(lower=1)
        df["pres_gap_frac"] = (1.0 - n_days / live_span).clip(lower=0.0)
        return df
