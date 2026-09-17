"""Bayram-clean the month-of-year aggregates.

THE DEFECT, same class as the sdly_364 misalignment that arm H fixed, on features H did not touch.
Bayram moves ~11 days earlier each solar year, so a feast lands in different calendar months in
different years. Month-of-year means built over history therefore absorb a feast that the month
they are applied to does not contain.

MEASURED on the industrial quartile (weekday mean > weekend mean, n=1,533), 2025 history:

    month 3 (Ramazan 2025, Mar 30-Apr 1)   6.8324 with bayram   6.8548 without   bias -0.0224
    month 4                                6.7781               6.7849           bias -0.0068
    month 6 (Kurban 2025, Jun 6-9)         7.0684               7.1172           bias -0.0488

Kurban 2026 is May 27-30. So `tan_moy_mean[6]` arrives at June 2026 carrying a -0.0488 dip for a
month that has no feast in it, while month 5's clean history is applied to a May that does. June is
30 of the 122 test days.

REACHES COLD START. `ilce_moy_mean` and `ilce_lf_moy` are in BASE_FEATS, the set an unseen row
depends on -- so this is a repair aimed at the segment carrying 65% of the MSE, and it adds no
information, it removes a contaminant.

WHAT IS AND IS NOT VALIDATABLE. With 15 months of data the only feast whose month-of-year source
is itself in history is Ramazan: the winter fold (val Dec 2025 - Mar 2026) contains Ramazan 2026
and its month-3 source sits in 2025. Kurban's June contamination CANNOT be validated on any fold --
the summer folds validate Apr-Jul 2025, whose month-of-year sources are in 2024 and absent. The
two-direction test is therefore unavailable here, and this is offered as a defect repair with a
measured mechanism, not as a validated gain. Said plainly because a winter-only win would otherwise
look like evidence it is not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# feast windows present in the data, plus arefe (the eve), which carries its own distinct profile
BAYRAM_DAYS = pd.DatetimeIndex(np.concatenate([
    pd.date_range("2025-03-29", "2025-04-01").values,   # Ramazan 2025 + arefe
    pd.date_range("2025-06-05", "2025-06-09").values,   # Kurban 2025 + arefe
    pd.date_range("2026-03-19", "2026-03-22").values,   # Ramazan 2026 + arefe
    pd.date_range("2026-05-26", "2026-05-30").values,   # Kurban 2026 + arefe (test window)
]))

MOY_KEYS = ("tm", "ilce_m", "ilce_lf_m")


def clean_moy(hist: pd.DataFrame, A: dict) -> dict:
    """Recompute the three month-of-year aggregates from non-feast days only.

    Everything else in `A` is left exactly as `build_aggs` produced it: the feast days still count
    toward tan_mean, the rolling windows and the day-of-week profile, where they belong. Only the
    MONTH axis is misaligned across years, so only the month axis is repaired.
    """
    h = hist[~hist.tarih.isin(BAYRAM_DAYS)]
    if len(h) == len(hist):
        return A                                    # no feast in this history slice; nothing to do

    out = dict(A)
    out["tm"] = (h.groupby(["tanim", "month"]).log_t.mean()
                 .rename("tan_moy_mean").reset_index())
    out["ilce_m"] = (h.groupby(["ilce", "month"]).log_t.mean()
                     .rename("ilce_moy_mean").reset_index())
    lf = h.assign(lf=h.tuketim / (h.guc.clip(lower=1) * 24))
    out["ilce_lf_m"] = (lf.groupby(["ilce", "month"]).lf.mean()
                        .rename("ilce_lf_moy").reset_index())
    return out


def contamination_report(hist: pd.DataFrame) -> pd.DataFrame:
    """Per-month bias that `clean_moy` removes, on the industrial quartile."""
    d = hist.assign(wknd=hist.tarih.dt.dayofweek.isin([5, 6]))
    prof = d.groupby(["tanim", "wknd"]).log_t.mean().unstack()
    ind = prof.index[(prof.get(False) - prof.get(True)) > 0]
    t = hist[hist.tanim.isin(set(ind))].assign(
        is_b=hist.tarih.isin(BAYRAM_DAYS), moy=hist.tarih.dt.month)
    g = t.groupby("moy")
    return pd.DataFrame({
        "rows": g.size(),
        "bayram_rows": g.is_b.sum(),
        "with_bayram": g.log_t.mean(),
        "without_bayram": t[~t.is_b].groupby("moy").log_t.mean(),
    }).assign(bias=lambda d: d.with_bayram - d.without_bayram)
