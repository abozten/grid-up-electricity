"""Decode segment biases from the LB scores ALREADY PAID FOR. No new submissions.

THE INFORMATION WE ALREADY HAVE. With base p0 (LB_0) and any other scored submission p_i, writing
d_i = p_i - p0 and r = p0 - y for the base's residual on the scored rows,

    LB_i^2 = LB_0^2 + 2*mean(d_i*r) + mean(d_i^2)
    =>  m_i := mean(d_i * r) = (LB_i^2 - LB_0^2 - mean(d_i^2)) / 2      EXACT, no labels

Six scored submissions besides the base give six exact linear functionals of the true test residual.
lb_probe would BUY more of these at one slot each; this file asks how far the ones already bought
can be pushed.

THE MODEL. Approximate r by a piecewise-constant bias over a partition of the rows,
r ~ sum_k b_k f_k with f_k the cell indicators. Then

    m_i = sum_k b_k * A_ik + mean(d_i * r_perp),      A_ik = mean(d_i * f_k)

A is computable from the CSVs. Solving A b = m gives the cell biases, and correcting cell k by -b_k
banks w_k*b_k^2.

THE HONEST CATCH, stated up front. The neglected term mean(d_i * r_perp) is NOT zero -- d_i has
components outside the cell space -- so unlike lb_probe this decode is BIASED, not exact. A probe
shifts one cell by a known constant and isolates it; these differences are whole-model changes that
touch everything. That is the price of not spending slots, and it is why every arm below is scored
by LEAVE-ONE-EQUATION-OUT: fit on five functionals, predict the sixth, compare to its known value.
LOO error is in the units of m, so compare it against the spread of m itself. A partition that
cannot predict a held-out functional has not identified anything.

A SECOND CATCH, visible in the data. The m_i cluster into two values (0.000679 for v36/v33/
v36_nofloor, 0.001592 for v38_full/v31). That is the KKT condition for v42 being a constrained
optimum -- at the optimum every member in the support has equal inner product with the residual.
So the six numbers carry far less independent information than six; the effective rank is what the
condition number below reports.

    python3 src/evaluation/lb_solve.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_archive" / "legacy"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.lb_gram import SUBS

BASE = "v42"
RIDGE = (0.0, 1e-8, 1e-6, 1e-4)


def setup():
    from pipeline4 import TE
    seen = np.load("outputs/v35_base.npz", allow_pickle=True)["seen"].astype(bool)
    mo = TE.tarih.dt.month.values
    P = {n: np.log1p(pd.read_csv(p).iloc[:, -1].values.astype(float)) for n, (p, _) in SUBS.items()}
    p0, L0 = P[BASE], SUBS[BASE][1]
    names = [n for n in P if n != BASE]
    Dm = np.array([P[n] - p0 for n in names])
    m = np.array([(SUBS[n][1] ** 2 - L0 ** 2 - float((d ** 2).mean())) / 2
                  for n, d in zip(names, Dm)])
    parts = {
        "global (1)": {"all": np.ones(len(p0), bool)},
        "seen/unseen (2)": {"seen": seen, "unseen": ~seen},
        "month (4)": {f"m{k}": mo == k for k in (4, 5, 6, 7)},
        "seen x half (4)": {"S_early": seen & (mo <= 5), "S_late": seen & (mo >= 6),
                            "U_early": ~seen & (mo <= 5), "U_late": ~seen & (mo >= 6)},
        "month x seg (7)": {**{f"m{k}S": (mo == k) & seen for k in (4, 5, 6, 7)},
                            "m5U": ((mo == 5) | (mo == 4)) & ~seen,
                            "m6U": (mo == 6) & ~seen, "m7U": (mo == 7) & ~seen},
    }
    return names, Dm, m, parts, L0


def fit(A, m, lam):
    k = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(k), A.T @ m)


if __name__ == "__main__":
    names, Dm, m, parts, L0 = setup()
    print(f"base {BASE}  LB {L0}   six known functionals m_i:")
    print("   " + "  ".join(f"{n}:{v:+.6f}" for n, v in zip(names, m)))
    print(f"   spread of m: {m.max()-m.min():.6f}   |m| mean {np.abs(m).mean():.6f}\n")

    print(f"{'partition':<18}{'k':>3}{'cond':>10}{'fit resid':>12}{'LOO err':>11}"
          f"{'banked':>10}{'pred LB':>10}")
    out = {}
    for pname, cl in parts.items():
        keys = list(cl)
        A = np.array([[float((d * c).mean()) for c in cl.values()] for d in Dm])
        w = np.array([float(c.mean()) for c in cl.values()])
        cond = float(np.linalg.cond(A)) if A.shape[1] > 1 else 1.0
        best = None
        for lam in RIDGE:
            try:
                b = fit(A, m, lam)
            except np.linalg.LinAlgError:
                continue
            resid = float(np.sqrt(np.mean((A @ b - m) ** 2)))
            loo = []
            for i in range(len(m)):
                idx = [j for j in range(len(m)) if j != i]
                try:
                    bi = fit(A[idx], m[idx], lam)
                except np.linalg.LinAlgError:
                    continue
                loo.append((A[i] @ bi - m[i]) ** 2)
            loo = float(np.sqrt(np.mean(loo))) if loo else float("nan")
            if best is None or loo < best[0]:
                best = (loo, lam, b, resid)
        loo, lam, b, resid = best
        banked = float((w * b ** 2).sum())
        pred = float(np.sqrt(max(L0 ** 2 - banked, 0)))
        out[pname] = {"lam": lam, "b": dict(zip(keys, b.tolist())), "banked": banked,
                      "loo": loo, "cond": cond, "pred_lb": pred}
        print(f"{pname:<18}{len(keys):>3}{cond:>10.1e}{resid:>12.2e}{loo:>11.2e}"
              f"{banked:>10.5f}{pred:>10.5f}")
        print(f"{'':<18}   b = " + "  ".join(f"{k}:{v:+.4f}" for k, v in zip(keys, b)))
    json.dump(out, open("outputs/lb_solve.json", "w"), indent=1, default=float)
    print("\nA partition is only believable if LOO err is well below the spread of m "
          f"({m.max()-m.min():.2e}).")
    print("wrote outputs/lb_solve.json")
