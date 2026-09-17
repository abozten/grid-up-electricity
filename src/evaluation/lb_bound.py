"""Pre-submission LB interval, anchored on a submission whose LB score is KNOWN.

WHY THIS EXISTS. A level-aligned local fold is IMPOSSIBLE with this data:

    Apr-Jul 2025  labels yes, but only 3.0 months of prior history
    Apr-Jul 2026  15.0 months of history, but labels are the hidden test set

The scored window needs BOTH season and 15 months of history, and no fold has both. That is why
every local score sits 0.09 below the LB and why fold DELTAS have transferred at 80%, 10%,
NEGATIVE and CONFIRMED. Local levels cannot be aligned. Stop trying.

WHAT CAN BE DONE INSTEAD. RMSLE is a quadratic in the predictions, so with one known LB anchor the
candidate's LB is pinned to an exact interval that needs no labels at all. With d = log1p(cand) -
log1p(anchor) over the N test rows:

    dMSE = (1/N) * sum[ d^2 ] - (2/N) * sum[ d * (y - log1p(anchor)) ]
             ^^^^^^^^^^^^^^^                    ^^^^^^^^^^^^^^^^^^^^
             D, known exactly                   unknown, |.| bounded by Cauchy-Schwarz

so  dMSE in [ D - 2*sqrt(D*R), D + 2*sqrt(D*R) ]  where R = anchor's own MSE = LB_anchor^2.
That Cauchy-Schwarz interval is the ONLY hard bound, and it is useless in practice (+/-0.04).

[-D, +D] IS NOT A BOUND. I initially wrote it as one. The v37 retro falsified it: actual dMSE was
159% of D, outside +D, because the panel features moved predictions ANTI-correlated with the
anchor's residual -- past the truth and out the other side. [-D,+D] holds only if the change is
non-negatively correlated with the error, which is an assumption, not a guarantee. The v36_nofloor
interval I quoted the night before was therefore lucky, not sound.

The interpretable corners below are SCENARIOS, not bounds:

    candidate exactly right on the rows it moves   dMSE = -D
    anchor exactly right on those rows             dMSE = +D
    those rows are truly zero                      dMSE = (1/N) sum[ cand^2 - anchor^2 ]

THE NULL IS NOT ZERO -- IT IS +D. A change uncorrelated with the anchor's error costs +D. To break
even the change must be positively correlated with the anchor's residual, and it must clear
sum[d*(y-anchor)] = sum[d^2]/2. Bigger changes need proportionally more correlation to pay off.
This reframes "is my change good" as "have I earned the D I am spending", which is measurable.

RETRO on the three pairs with two known LB scores each:

    change        D (null cost)   actual dMSE   earned / needed
    v36           0.006530          -0.000889     114%   <- paid for itself
    v36_nofloor   0.001631          +0.001323      19%
    v37_full      0.002124          +0.003371     -59%   <- anti-correlated

D is exact and knowable BEFORE submitting. The earned fraction is not. So this tool sizes the
STAKES of a change, and cannot give its SIGN. Treat a large D as a large bet, not as a large gain.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

N_TEST = 714688


def load(path):
    d = pd.read_csv(path)
    return np.log1p(d.iloc[:, -1].values.astype(float)), d.iloc[:, 0].values


def report(anchor_path, cand_path, lb_anchor, name="", actual=None):
    a, ida = load(anchor_path)
    c, idc = load(cand_path)
    assert (ida == idc).all(), "id order differs"
    N = len(a)
    d = c - a
    S = d != 0
    D = float((d ** 2).sum() / N)
    zero_case = float(((c ** 2 - a ** 2)[S]).sum() / N)
    R = lb_anchor ** 2
    lo_cs, hi_cs = D - 2 * np.sqrt(D * R), D + 2 * np.sqrt(D * R)

    def lb(dmse):
        v = R + dmse
        return float(np.sqrt(v)) if v > 0 else float("nan")

    print(f"\n=== {name or cand_path} vs anchor {anchor_path.split('/')[-1]} (LB {lb_anchor:.5f}) ===")
    print(f"  rows moved {S.sum():,} ({100*S.mean():.3f}%)   mean |d| on moved {np.abs(d[S]).mean():.4f}")
    print(f"  D = mean d^2 = {D:.6f}   <- the NULL cost: an uncorrelated change pays this")
    print(f"    candidate right on moved rows   dMSE {-D:+.5f}   LB {lb(-D):.5f}")
    print(f"    moved rows truly zero           dMSE {zero_case:+.5f}   LB {lb(zero_case):.5f}")
    print(f"    anchor right on moved rows      dMSE {+D:+.5f}   LB {lb(D):.5f}")
    print(f"    NULL (change uncorrelated)      dMSE {+D:+.5f}   LB {lb(D):.5f}")
    print(f"  hard bound (Cauchy-Schwarz)       LB [{lb(lo_cs):.5f}, {lb(hi_cs):.5f}]  (useless, but true)")
    print(f"  TO BREAK EVEN the change must earn sum[d*(y-anchor)]/N = D/2 = {D/2:+.6f}")
    if actual is not None:
        dm = actual ** 2 - R
        # implied correlation earned
        earned = (D - dm) / 2
        print(f"  ACTUAL LB {actual:.5f}  (dMSE {dm:+.6f})  -> earned {earned:+.6f} of the "
              f"{D/2:+.6f} needed to break even  = {100*earned/(D/2):.0f}%")
        inside = -D - 1e-9 <= dm <= D + 1e-9
        print(f"  inside [-D,+D]? {'yes' if inside else 'NO -- anti-correlated with the error'}")
        return dm, D
    return None, D


if __name__ == "__main__":
    if len(sys.argv) > 2:
        report(sys.argv[1], sys.argv[2], float(sys.argv[3]),
               name=sys.argv[4] if len(sys.argv) > 4 else "")
    else:
        print("RETROACTIVE VALIDATION -- every pair where both LB scores are known")
        O = "outputs/"
        cases = [(O+"submission_v36.csv", O+"submission_v37_full.csv", 1.03333, "v37_full", 1.03496),
                 (O+"submission_v36.csv", O+"submission_v36_nofloor.csv", 1.03333, "v36_nofloor", 1.03397),
                 (O+"submission_v33.csv", O+"submission_v36.csv", 1.03376, "v36", 1.03333)]
        rows = []
        for ap, cp, lba, nm, act in cases:
            dm, D = report(ap, cp, lba, nm, act)
            rows.append((nm, D, dm))
        print("\n=== summary: did the +/-D interval bracket every known outcome? ===")
        print(f"{'change':<14}{'D (null cost)':>15}{'actual dMSE':>14}{'as % of D':>12}  bracketed")
        for nm, D, dm in rows:
            print(f"{nm:<14}{D:>15.6f}{dm:>+14.6f}{100*dm/D:>11.0f}%  "
                  f"{'YES' if abs(dm) <= D + 1e-9 else 'NO'}")
