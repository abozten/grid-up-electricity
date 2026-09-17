"""Causal walk-forward OOF harness with a pre-registered gate. The north star for every candidate.

WHY THIS EXISTS. Every verdict in this project has come from a SINGLE holdout window, and single
windows have repeatedly lied here:

  v20        won on LightGBM, reversed at the blend                     (reports/V20.md)
  KGUP       -0.0086 on autumn, entirely an artefact of refitting the
             blend weight on the fold being scored                      (reports/AUG24_LEDGER.md)
  ytp        -0.0034 winter, +0.0015 autumn                             (reports/AUG24_LEDGER.md)
  H, W=10    -0.0074 on LightGBM winter, +0.0009 at the blend, while
             the leaderboard says the shipped version GAINS             (outputs/wsweep.json)

A number from one window is a hypothesis, not a result.

WHAT OOF MEANS FOR A CAUSAL PANEL. Not k-fold: shuffling rows across time would leak the future
into the past. Each fold trains on everything strictly before its cutoff and predicts one forward
window; the windows tile the timeline without overlapping, so concatenating them yields one
out-of-fold prediction for (almost) every row, each made by a model that never saw it or anything
after it. Candidates are then scored on the POOLED OOF vector -- same rows, same weighting, every
time -- instead of on whichever window happened to be run.

WHAT THE GATE FIXES. Acceptance criteria are evaluated by `gate()` from a spec fixed BEFORE results
are seen, not applied by eye afterwards. Applying them afterwards is how v14, v17 and v20 each got
talked into being submitted.

UNINFORMATIVE FOLDS. A fold where the candidate provably cannot change any input is not evidence
against it -- it is noise wearing the costume of evidence. The bayram-axis fix H is the worked
example: of three folds, 2025-09 contains no feast at all and 2025-06 contains feasts whose
lag-364 sources predate the dataset, so H alters nothing in either. Pooling them and then reading
"reverses on all three folds" measures seed noise twice and the treatment once. Pass
`informative=[...]` to Gate so those folds are excluded from the consistency check; they still
count in the pooled score, where their contribution is genuinely neutral.

HORIZON MATTERS. Test predicts 1-122 days ahead from a single cutoff. A fold with a 30-day window
tests a much easier problem, so `pooled()` reports error banded by horizon as well as by segment --
a candidate that wins at short horizon and loses past day 60 is not a candidate for this test set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

S_TEST = 0.2216          # unseen share of the real test set; folds run 7-14%, so reweight always
SEED_SPREAD = 0.015      # documented run-to-run spread; nothing below a third of this is a result
HORIZON_BANDS = [(1, 30), (31, 60), (61, 90), (91, 122)]

# Cutoffs tile Jun 2025 - Mar 2026 without overlap. Each trains on everything before `cut`.
# The last fold is the most test-like (deepest history, longest window) and is listed last so a
# partial run still ends on the informative one.
FOLDS = [("2025-06-01", "2025-09-01"),
         ("2025-09-01", "2025-12-01"),
         ("2025-12-01", "2026-04-01")]


def rmsle(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@dataclass
class OOF:
    """Pooled out-of-fold predictions for one candidate."""
    name: str
    y: np.ndarray
    pred: np.ndarray
    seen: np.ndarray
    horizon: np.ndarray
    fold: np.ndarray

    def segments(self) -> dict:
        s, u = self.seen.astype(bool), ~self.seen.astype(bool)
        out = {"all": rmsle(self.pred, self.y),
               "seen": rmsle(self.pred[s], self.y[s]),
               "unseen": rmsle(self.pred[u], self.y[u]),
               "n": int(len(self.y)), "unseen_share": float(u.mean())}
        # reweight to the test mix: fold composition is not test composition and never will be
        out["test_comp"] = float(np.sqrt((1 - S_TEST) * out["seen"] ** 2 + S_TEST * out["unseen"] ** 2))
        for lo, hi in HORIZON_BANDS:
            m = (self.horizon >= lo) & (self.horizon <= hi)
            if m.any():
                out[f"h{lo}_{hi}"] = rmsle(self.pred[m], self.y[m])
        for f in np.unique(self.fold):
            m = self.fold == f
            out[f"fold_{f}"] = rmsle(self.pred[m], self.y[m])
        return out


@dataclass
class Gate:
    """Acceptance criteria. Instantiate and print BEFORE fitting anything."""
    min_gain: float = SEED_SPREAD / 3      # 0.005 on test_comp
    max_segment_regression: float = 0.002  # neither seen nor unseen may worsen by more than this
    require_all_folds: bool = True         # per-fold delta must not reverse sign on any fold
    require_all_horizons: bool = True      # nor may it reverse in any horizon band
    informative: tuple = ()                # folds that can express the treatment; () means all
    notes: str = ""

    def describe(self) -> str:
        scope = ("all folds" if not self.informative
                 else f"informative folds {list(self.informative)}")
        return (f"GATE  gain > {self.min_gain:.4f} on test_comp | no segment worse than "
                f"{self.max_segment_regression:+.4f} | consistent across "
                f"{scope if self.require_all_folds else 'pooled only'}"
                f"{' and all horizon bands' if self.require_all_horizons else ''}"
                + (f" | {self.notes}" if self.notes else ""))

    def evaluate(self, base: OOF, cand: OOF) -> dict:
        b, c = base.segments(), cand.segments()
        d = c["test_comp"] - b["test_comp"]
        reasons = []
        if d > -self.min_gain:
            reasons.append(f"gain {d:+.4f} does not clear the {self.min_gain:.4f} floor")
        for seg in ("seen", "unseen"):
            if c[seg] - b[seg] > self.max_segment_regression:
                reasons.append(f"{seg} regresses {c[seg]-b[seg]:+.4f}")
        if self.require_all_folds:
            fk = [k for k in b if k.startswith("fold_")]
            if self.informative:
                fk = [k for k in fk if k[len("fold_"):] in self.informative]
            if not fk:
                reasons.append("no informative fold in this run -- INCONCLUSIVE, not a rejection")
            bad = [k for k in fk if (c[k] - b[k]) > 0]
            if bad:
                reasons.append(f"reverses on {', '.join(bad)}")
        if self.require_all_horizons:
            bad = [k for k in b if k.startswith("h") and k[1].isdigit() and (c[k] - b[k]) > 0]
            if bad:
                reasons.append(f"reverses in horizon band(s) {', '.join(bad)}")
        return {"delta_test_comp": d, "passes": not reasons,
                "reasons": reasons or ["clears every criterion"],
                "base": b, "candidate": c}


def report(base: OOF, cand: OOF, gate: Gate) -> dict:
    r = gate.evaluate(base, cand)
    b, c = r["base"], r["candidate"]
    print(f"\n{gate.describe()}")
    print(f"\n  {'metric':14s} {base.name:>10s} {cand.name:>10s} {'delta':>9s}")
    keys = ["test_comp", "all", "seen", "unseen"] + \
           [k for k in b if k.startswith("h") and k[1].isdigit()] + \
           [k for k in b if k.startswith("fold_")]
    for k in keys:
        print(f"  {k:14s} {b[k]:10.4f} {c[k]:10.4f} {c[k]-b[k]:+9.4f}")
    print(f"\n  VERDICT: {'PASS' if r['passes'] else 'REJECT'} -- " + "; ".join(r["reasons"]))
    return r
