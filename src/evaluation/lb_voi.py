"""What is a submission slot WORTH? Value-of-information over the LB Gram, with no labels.

`lb_gram.py` recovers the exact residual Gram of submissions whose LB is known and blends them
optimally. This asks the forward question: of the artifacts already built but never scored, which
one should get the next slot?

Ranking them by their own expected LB is the wrong objective. What a new member contributes to the
BLEND is its geometry -- a candidate whose predictions are an affine combination of members we have
already scored adds no new direction, so the optimum cannot move no matter what it returns. A
candidate that moves orthogonally to the span adds a genuinely new axis. Both facts are computable
with no labels.

TWO THINGS THIS COMPUTES, both exact.

1. THE FEASIBLE LB INTERVAL, from every known member at once. With residual Gram G over k scored
   submissions and D_i = mean((p_c - p_i)^2) known for each, hypothesising G_cc = t fixes the whole
   augmented Gram, because

       <r_c, r_i> = (t + G_ii - D_i)/2                                     (exact, affine in t)

   A Gram matrix must be PSD, so t >= g(t)' G^-1 g(t). With h_i = G_ii - D_i that reduces to

       a*t^2 + (2b - 4)*t + c <= 0,   a = 1'G^-1 1,  b = 1'G^-1 h,  c = h'G^-1 h

   a convex parabola in t, so the feasible set is the interval between its roots. LB_c must land in
   [sqrt(t_lo), sqrt(t_hi)] -- a hard bound, before submitting, from no labels.

   With ONE anchor this reduces exactly to the triangle inequality |sqrt(R) - sqrt(D)| <= LB_c <=
   sqrt(R) + sqrt(D), which is the Cauchy-Schwarz interval `lb_bound.py` reports and correctly
   calls useless (+/-0.04). Every extra scored member tightens it, and the retro below measures by
   how much. This does NOT supersede lb_bound: that tool sizes the STAKE of a change against one
   anchor and explains what must be earned. This one brackets the OUTCOME using all six.

2. THE VOI CURVE. For each hypothesised t across the feasible interval, re-solve the optimal blend
   over {scored members} + {candidate} and report the predicted blend LB. The candidate's value is
   how much that improves on the blend we can already build -- read across the whole interval, not
   at one guess, since the return is not known.

REDUNDANCY IS THE HEADLINE. v39 and v42 ARE blends of already-scored submissions, so they sit in
the span by construction: their feasible interval collapses to a point and their VOI is zero. That
is not a criticism of them as submissions -- a collapsed interval means the Gram PREDICTS their LB
exactly, so scoring one is a clean falsification test of the whole framework. It does mean they
cannot teach the blend anything new, and that a slot spent on one buys validation, not geometry.

THE STANDING RISK, unchanged from lb_gram: LB scores are the PUBLIC subset, so all of this is
public-row geometry and the private score may differ.

    retro:   python3 src/evaluation/lb_voi.py retro
    rank:    python3 src/evaluation/lb_voi.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.lb_gram import SUBS, gram, solve

CAND = {  # built, never scored -- the real decision set
    "v42_blend25":    "outputs/submission_v42_blend25.csv",
    "v41_segshift":   "outputs/submission_v41_segshift.csv",
    "v43_unseenzero": "outputs/submission_v43_unseenzero.csv",
    "v39_lbblend":    "outputs/submission_v39_lbblend.csv",
    "v40_unseenlift": "outputs/submission_v40_unseenlift.csv",
    "v35":            "outputs/submission_v35.csv",
    "v34":            "outputs/submission_v34.csv",
    "v32":            "outputs/submission_v32.csv",
    "v30":            "outputs/submission_v30.csv",
    "v35_ref":        "outputs/submission_v35_ref.csv",
    "v37_lean":       "outputs/submission_v37_lean.csv",
    "v37_noseas":     "outputs/submission_v37_noseas.csv",
    "v37_nofloor":    "outputs/submission_v37_nofloor.csv",
    "v38":            "outputs/submission_v38.csv",
    "v38_lean":       "outputs/submission_v38_lean.csv",
    "v38_noseas":     "outputs/submission_v38_noseas.csv",
    "v38_nofloor":    "outputs/submission_v38_nofloor.csv",
}
NGRID = 41


def logp(path):
    d = pd.read_csv(path)
    return np.log1p(d.iloc[:, -1].values.astype(float)), d.iloc[:, 0].values


def load_many(paths):
    P, ids = [], None
    for p in paths:
        v, i = logp(p)
        if ids is None:
            ids = i
        assert (i == ids).all(), f"{p}: id order differs"
        P.append(v)
    return np.array(P)


def feasible_t(G, h):
    """PSD-feasible interval for G_cc.  a t^2 + (2b-4) t + c <= 0."""
    Gi = np.linalg.inv(G)
    one = np.ones(len(G))
    a = float(one @ Gi @ one)
    b = float(one @ Gi @ h)
    c = float(h @ Gi @ h)
    A, B, C = a, 2 * b - 4, c
    disc = B * B - 4 * A * C
    if disc < 0:                      # numerically collapsed: the vertex is the only point
        return -B / (2 * A), -B / (2 * A)
    r = np.sqrt(disc)
    return (-B - r) / (2 * A), (-B + r) / (2 * A)


def augment(G, h, t):
    k = len(G)
    g = (t * np.ones(k) + h) / 2
    M = np.zeros((k + 1, k + 1))
    M[:k, :k] = G
    M[:k, k] = M[k, :k] = g
    M[k, k] = t
    return M


def span_fit(P, pc):
    """Best affine combination (weights summing to 1) of P reproducing pc, and what is left over.

    The leftover is the only part that can add a new axis to the blend; its RMS is exactly what
    widens the feasible interval, so the two diagnostics are the same fact seen twice.
    """
    ref = P[0]
    X = (P[1:] - ref).T
    y = pc - ref
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss = float((y ** 2).sum())
    w = np.empty(len(P))
    w[1:] = coef
    w[0] = 1 - coef.sum()
    r2 = 1 - float((resid ** 2).sum()) / ss if ss > 0 else 1.0
    return w, r2, float(np.sqrt((resid ** 2).mean()))


def blend_lb(G):
    w = solve(G)
    return float(np.sqrt(w @ G @ w)), w


def analyse(name, path, names, G, P, base_lb):
    pc = logp(path)[0]
    assert len(pc) == P.shape[1], f"{name}: row count differs"
    D = np.array([float(((pc - P[i]) ** 2).mean()) for i in range(len(names))])
    h = np.diag(G) - D
    t_lo, t_hi = feasible_t(G, h)
    lo, hi = np.sqrt(max(t_lo, 0)), np.sqrt(max(t_hi, 0))
    w_span, r2, orth = span_fit(P, pc)
    t_span = float(w_span @ G @ w_span)

    ts = np.linspace(t_lo, t_hi, NGRID) if t_hi > t_lo else np.array([t_lo])
    curve = []
    for t in ts:
        lb_b, w = blend_lb(augment(G, h, t))
        solo = float(np.sqrt(max(t, 0)))
        # SYNERGY is the only honest measure of what the blend gains. Beating `base_lb` is not
        # enough -- a candidate that returns 1.020 makes the blend 1.020 by taking all the weight,
        # which is the candidate being good, not the blend learning anything. Synergy asks how far
        # the blend beats the BETTER of {what we can already build, the candidate alone}.
        curve.append((solo, lb_b, float(w[-1]), min(base_lb, solo) - lb_b))

    # the cleanest single number: if it returns exactly the current best blend score, what does
    # adding it buy? That holds skill fixed and isolates decorrelation.
    lb_par, _ = blend_lb(augment(G, h, base_lb ** 2))
    par_gain = base_lb - lb_par if t_lo <= base_lb ** 2 <= t_hi else float("nan")

    # ceiling on decorrelation from the nearest scored member: at equal MSE, rho = 1 - D/(2R)
    i0 = int(np.argmin(D))
    rho_par = 1 - float(D[i0]) / (2 * float(G[i0, i0]))

    syn = [c[3] for c in curve]
    k = int(np.argmax(syn))
    return {"name": name, "D_min": float(D.min()), "lb_lo": lo, "lb_hi": hi,
            "width": hi - lo, "span_r2": r2, "orth_rms": orth, "rho_par": rho_par,
            "span_lb": float(np.sqrt(max(t_span, 0))), "curve": curve,
            "syn_max": syn[k], "syn_at": curve[k][0], "par_gain": par_gain,
            "w_cand_max": max(c[2] for c in curve)}


def retro(names, G, P):
    """Leave one scored submission out, bracket it from the other five, check it lands inside."""
    print("=== RETRO: bracket each KNOWN submission from the other five ===")
    print(f"{'held out':<14}{'actual LB':>11}{'interval from other 5':>26}"
          f"{'width':>9}{'1-anchor width':>16}  in?")
    ok = 0
    for j, nm in enumerate(names):
        idx = [i for i in range(len(names)) if i != j]
        Gs = G[np.ix_(idx, idx)]
        pc = P[j]
        D = np.array([float(((pc - P[i]) ** 2).mean()) for i in idx])
        h = np.diag(Gs) - D
        t_lo, t_hi = feasible_t(Gs, h)
        lo, hi = np.sqrt(max(t_lo, 0)), np.sqrt(max(t_hi, 0))
        act = SUBS[nm][1]
        inside = lo - 1e-6 <= act <= hi + 1e-6
        ok += inside
        # the single-anchor (lb_bound) interval, using the nearest anchor -- the fairest one
        b = int(np.argmin(D))
        R = Gs[b, b]
        w1 = (np.sqrt(R) + np.sqrt(D[b])) - abs(np.sqrt(R) - np.sqrt(D[b]))
        print(f"{nm:<14}{act:>11.5f}   [{lo:.5f}, {hi:.5f}]{hi-lo:>9.5f}{w1:>16.5f}  "
              f"{'yes' if inside else 'NO'}")
    print(f"\nbracketed {ok} of {len(names)}.  A miss would falsify the PSD argument or the "
          f"row alignment; nothing else can cause one.")


def main():
    names = list(SUBS)
    P = load_many([SUBS[n][0] for n in names])
    G = gram(names, P)
    base_lb, w0 = blend_lb(G)
    print(f"scored members: {', '.join(names)}")
    print(f"best blend available NOW: {base_lb:.5f}  "
          f"({' '.join(f'{n}:{x:.3f}' for n, x in zip(names, w0) if x > 1e-6)})\n")

    if len(sys.argv) > 1 and sys.argv[1] == "retro":
        retro(names, G, P)
        return

    rows = [analyse(nm, p, names, G, P, base_lb) for nm, p in CAND.items() if Path(p).exists()]
    rows.sort(key=lambda r: -r["syn_max"])

    print("=== candidates: what a slot on each one buys ===")
    print("rho@par = residual correlation with the nearest scored member IF it scores the same;")
    print("synergy = how far the blend beats the better of {best blend now, candidate alone}.\n")
    print(f"{'candidate':<17}{'feasible LB':>22}{'span R2':>9}{'rho@par':>9}"
          f"{'synergy':>10}{'  at return':>12}{'gain@par':>10}")
    for r in rows:
        pg = "  n/a" if np.isnan(r["par_gain"]) else f"{r['par_gain']:+.5f}"
        print(f"{r['name']:<17}[{r['lb_lo']:.5f}, {r['lb_hi']:.5f}]{r['span_r2']:>9.5f}"
              f"{r['rho_par']:>9.5f}{r['syn_max']:>+10.5f}{r['syn_at']:>12.5f}{pg:>10}")

    print("\n=== VOI curves: if it returns X, the blend can reach Y ===")
    for r in rows:
        if r["width"] < 1e-6:
            print(f"\n{r['name']}: IN SPAN (R2 {r['span_r2']:.6f}) -- the Gram predicts its LB as "
                  f"{r['span_lb']:.5f} and the blend cannot move. Scoring it TESTS the framework; "
                  f"it cannot extend it.")
            continue
        print(f"\n{r['name']}: orthogonal RMS {r['orth_rms']:.5f}, "
              f"max weight it would take {r['w_cand_max']:.3f}")
        pts = [r["curve"][i] for i in (0, NGRID // 4, NGRID // 2, 3 * NGRID // 4, NGRID - 1)]
        print("   " + "  ".join(f"{lb:.5f}->{bl:.5f}" for lb, bl, _, _ in pts))
        good = [lb for lb, bl, _, _ in r["curve"] if base_lb - bl > 1e-5]
        if good:
            print(f"   beats the current blend iff it returns better than {max(good):.5f}")

    # --- the frontier: what would a DECORRELATED submission be worth, at identical skill? ---
    # Every candidate above sits at rho > 0.998 with its nearest scored neighbour, because every
    # one of them is the same pipeline with a knob moved or a shift bolted on. Two predictors of
    # equal MSE and correlation rho blend to sigma*sqrt((1+rho)/2), so the value of a new member
    # is set almost entirely by rho, and barely at all by its own score. This prices that.
    print("\n=== DECORRELATION FRONTIER: a candidate that merely TIES the current blend ===")
    # A second model earns positive weight iff cov < sigma1^2, i.e. rho*s1*s2 < s1^2, i.e.
    # s2 < s1/rho. So decorrelation buys TOLERANCE for a worse solo score, and that bar is far
    # looser than "must beat the blend" -- which is the assumption every rejection here has used.
    print(f"{'rho vs blend':>13}{'blend LB if it ties':>21}{'gain':>10}"
          f"{'still helps up to':>19}   note")
    for rho in (0.999, 0.998, 0.995, 0.99, 0.98, 0.95, 0.90):
        lb = base_lb * np.sqrt((1 + rho) / 2)
        note = ""
        if rho >= 0.998:
            note = "<- where every existing artifact sits"
        elif rho <= 0.95:
            note = "<- worth more than every tuning gain measured here"
        print(f"{rho:>13.3f}{lb:>21.5f}{base_lb - lb:>+10.5f}"
              f"{base_lb / rho:>19.5f}   {note}")
    print("\nobserved residual correlations among the six SCORED submissions:")
    dg = np.sqrt(np.diag(G)); C = G / np.outer(dg, dg)
    off = C[np.triu_indices(len(names), 1)]
    print(f"  min {off.min():.5f}   median {np.median(off):.5f}   max {off.max():.5f}")
    i, j = np.unravel_index(np.argmin(C + np.eye(len(names)) * 9), C.shape)
    print(f"  most decorrelated pair: {names[i]} / {names[j]} at rho {C[i,j]:.5f}")


if __name__ == "__main__":
    main()
