"""Error explainability: given a scored dataframe, answer "where is the error and why".

WHY THIS EXISTS. `src/evaluation/oof.py` answers a narrow question well -- does candidate
X beat baseline Y on the pre-registered segments (seen/unseen, horizon, fold)? It cannot
tell you *which transformers*, *which districts*, *which feature range* the error concentrates
in, or what a candidate changed for the worse in a segment nobody thought to check. That gap
has historically been closed by one-off scripts (`reports/OUTAGES.md`, `ZERO_MASS.md`,
`REGIME.md`, `PHYSICAL_LIMITS.md` each started as a bespoke groupby-and-eyeball script) --
each rebuilding the same three operations: rank segments by their share of total squared
error, check whether error is systematically biased along a feature axis, and pull the worst
individual rows for a human (or an agent) to read. This module makes those three operations
reusable, and adds a fourth that the one-off scripts never had: a principled A/B segment diff,
so "does this candidate help or hurt" can be answered per-segment instead of only in aggregate.

This is deliberately pipeline-agnostic: it operates on any DataFrame with a y column, a pred
column (both in log1p space, matching this repo's RMSLE-on-log-space convention -- see
`src/evaluation/metrics.py`), and whatever metadata/feature columns the caller already has.
It does not load data, train anything, or know about `Config`, `FeaturePipeline`, or which of
this repo's two parallel code paths (see README.md) produced the predictions. Any experiment
script that already builds a validation DataFrame can hand it here as the last step.

INTENDED USAGE (by a future agent debugging a regression or hunting the next feature):
    from src.evaluation.explain import build_report, render_markdown, compare

    report = build_report(val_df, y_col="log_t", pred_col="pred",
                           group_cols=["ilce", "weekday", "history_bucket"],
                           feature_cols=["sicaklik", "n_valid_28", "dispersion_28"],
                           id_cols=["tanim", "tarih"], name="v35_candidate")
    Path("reports/EXPLAIN_v35.md").write_text(render_markdown(report))

    # or, to see exactly what a change helped/hurt relative to the shipped baseline:
    diff = compare(base_df, cand_df, id_col="row_id", y_col="log_t",
                    pred_col="pred", group_cols=["ilce", "history_bucket"])

Every number here is plain arithmetic on already-scored predictions -- no model is fit, so
this is safe to run on anything, including the frozen OOF arrays already sitting in
`outputs/*.json` once they are reattached to their source rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A segment/bin with fewer rows than this is noise, not a finding -- the same spirit as
# oof.py's SEED_SPREAD gate, applied per-group instead of per-run.
DEFAULT_MIN_N = 30


def _rmsle(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")


def add_error_columns(df: pd.DataFrame, y_col: str, pred_col: str,
                       weight_col: str | None = None) -> pd.DataFrame:
    """Return a copy of `df` with `err` (signed, pred - y), `sq_err`, `abs_err` attached.

    `y_col`/`pred_col` are assumed to already be in log1p space, i.e. `err` is directly the
    per-row contribution to RMSLE -- this matches every OOF array already produced in this
    repo (see `src/evaluation/oof.py`, `src/evaluation/metrics.py`).
    """
    out = df.copy()
    out["err"] = out[pred_col].astype(float) - out[y_col].astype(float)
    if weight_col is not None:
        out["err"] = out["err"] * np.sqrt(out[weight_col].astype(float))
    out["sq_err"] = out["err"] ** 2
    out["abs_err"] = out["err"].abs()
    return out


def segment_report(df: pd.DataFrame, by: str | list[str], min_n: int = DEFAULT_MIN_N,
                    top: int = 20) -> pd.DataFrame:
    """Rank values of `by` by how much of the TOTAL squared error they explain.

    Two columns matter more than the segment's own RMSLE:
      - `share_of_error`: this segment's fraction of the sum of squared error across all rows.
      - `lift`: share_of_error / share_of_rows. lift > 1 means the segment is a
        disproportionate error source relative to its size -- the actionable signal. A
        segment with high RMSLE but 0.3% of rows is not worth chasing; one with lift 3x and
        15% of rows is where the next feature or routing fix should look.

    `df` must already carry `sq_err` (see `add_error_columns`). Requires `y`/`pred` NOT to be
    passed here -- this operates purely on the attached error columns so it composes with
    `compare()` below.
    """
    assert "sq_err" in df.columns, "call add_error_columns() first"
    total_sq = df["sq_err"].sum()
    n_total = len(df)
    g = df.groupby(by, observed=True)
    rep = g.agg(n=("sq_err", "size"), rmsle=("sq_err", lambda s: _rmsle(np.sqrt(s))),
                bias=("err", "mean"), sq_err_sum=("sq_err", "sum"))
    rep["share_of_rows"] = rep["n"] / n_total
    rep["share_of_error"] = rep["sq_err_sum"] / total_sq if total_sq > 0 else 0.0
    rep["lift"] = rep["share_of_error"] / rep["share_of_rows"].replace(0, np.nan)
    rep = rep.drop(columns="sq_err_sum")
    rep = rep[rep["n"] >= min_n]
    return rep.sort_values("share_of_error", ascending=False).head(top)


def bias_curve(df: pd.DataFrame, feature: str, bins: int = 10,
                min_n: int = DEFAULT_MIN_N) -> pd.DataFrame:
    """Bin `feature` into `bins` quantile buckets and report RMSLE/bias per bucket.

    Surfaces the pattern segment_report cannot: systematic miscalibration *along a
    continuous axis* -- e.g. bias that grows more negative as `n_valid_28` shrinks means
    under-shrinkage toward the anchor for thin-history transformers, independent of any
    single segment being flagged. `monotonic_bias` is True when the bin-bias sequence is
    non-decreasing or non-increasing end to end (ignoring bins dropped for low n) -- a
    monotonic trend is a specific, actionable hypothesis; a non-monotonic one usually is not.
    """
    assert "err" in df.columns, "call add_error_columns() first"
    work = df[[feature, "err", "sq_err"]].dropna(subset=[feature])
    try:
        work = work.assign(_bin=pd.qcut(work[feature], q=bins, duplicates="drop"))
    except ValueError:
        # too few unique values for the requested bin count
        work = work.assign(_bin=pd.cut(work[feature], bins=min(bins, work[feature].nunique())))
    rep = work.groupby("_bin", observed=True).agg(
        n=("err", "size"), feature_mean=(feature, "mean"),
        rmsle=("sq_err", lambda s: _rmsle(np.sqrt(s))), bias=("err", "mean"),
    )
    rep = rep[rep["n"] >= min_n].reset_index(drop=True)
    biases = rep["bias"].to_numpy()
    diffs = np.diff(biases)
    monotonic = bool(len(diffs) >= 2 and (np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12)))
    rep.attrs["feature"] = feature
    rep.attrs["monotonic_bias"] = monotonic
    return rep


def worst_cases(df: pd.DataFrame, id_cols: list[str], k: int = 25,
                 extra_cols: list[str] | None = None) -> pd.DataFrame:
    """The k rows with the largest absolute error, for a human/agent to read individually.

    Segment leaderboards find *where*; this is for *why*, one row at a time -- pair it with
    domain columns (weather, presence flags, history length) via `extra_cols` so the picture
    doesn't need a second query.
    """
    assert "abs_err" in df.columns, "call add_error_columns() first"
    cols = list(dict.fromkeys(id_cols + ["err", "abs_err"] + (extra_cols or [])))
    return df.nlargest(k, "abs_err")[cols].reset_index(drop=True)


@dataclass
class CompareResult:
    name: str
    delta_rmsle: float
    n: int
    by: list
    table: pd.DataFrame  # per-segment: rmsle_base, rmsle_cand, delta, n
    improved: pd.DataFrame
    regressed: pd.DataFrame


def compare(df_base: pd.DataFrame, df_cand: pd.DataFrame, id_col: str, y_col: str,
            pred_col: str, group_cols: list[str], min_n: int = DEFAULT_MIN_N,
            noise_floor: float = 0.005) -> CompareResult:
    """Per-segment A/B diff: exactly which segments a candidate helped or hurt.

    Aligns `df_base` and `df_cand` on `id_col` (so the two frames may be scored on the same
    rows in any order, or be different columns of the same experiment's val_block), then for
    every `group_cols` value reports base/candidate RMSLE and the delta. `noise_floor`
    defaults to a third of this repo's documented seed-to-seed spread (see oof.py's
    SEED_SPREAD) -- deltas smaller than that are not reported as improved/regressed, only in
    the full `table`.

    This is the tool for the question "the aggregate gate says PASS/REJECT, but WHERE did
    that number come from" -- a candidate can pass the aggregate gate while quietly making a
    minority segment much worse, and this is how that gets caught before it ships.
    """
    b = df_base[[id_col, y_col, pred_col] + group_cols].rename(
        columns={pred_col: "_pred_base"})
    c = df_cand[[id_col, pred_col]].rename(columns={pred_col: "_pred_cand"})
    m = b.merge(c, on=id_col, how="inner")
    assert len(m) > 0, "no overlapping ids between df_base and df_cand"
    m["sq_err_base"] = (m["_pred_base"] - m[y_col]) ** 2
    m["sq_err_cand"] = (m["_pred_cand"] - m[y_col]) ** 2

    g = m.groupby(group_cols, observed=True)
    table = g.agg(
        n=("sq_err_base", "size"),
        rmsle_base=("sq_err_base", lambda s: _rmsle(np.sqrt(s))),
        rmsle_cand=("sq_err_cand", lambda s: _rmsle(np.sqrt(s))),
    )
    table["delta"] = table["rmsle_cand"] - table["rmsle_base"]
    table = table[table["n"] >= min_n].sort_values("delta")

    overall_delta = _rmsle(np.sqrt(m["sq_err_cand"])) - _rmsle(np.sqrt(m["sq_err_base"]))
    improved = table[table["delta"] <= -noise_floor].sort_values("delta")
    regressed = table[table["delta"] >= noise_floor].sort_values("delta", ascending=False)
    return CompareResult(name=f"{'+'.join(group_cols)}", delta_rmsle=overall_delta, n=len(m),
                          by=group_cols, table=table, improved=improved, regressed=regressed)


def build_report(df: pd.DataFrame, y_col: str, pred_col: str, group_cols: list[str],
                  feature_cols: list[str] | None = None, id_cols: list[str] | None = None,
                  name: str = "run", min_n: int = DEFAULT_MIN_N, top_segments: int = 15,
                  k_worst: int = 20, bins: int = 10) -> dict:
    """Orchestrate segment/bias/worst-case analysis into one JSON-able payload.

    One call per candidate is the intended usage -- see module docstring. Returns plain
    dict/DataFrame-friendly structures (`render_markdown` below turns it into text; every
    DataFrame inside can also be `.to_json()`'d directly for `outputs/*.json`).
    """
    work = add_error_columns(df, y_col, pred_col)
    overall = {"n": len(work), "rmsle": _rmsle(work["err"].to_numpy()),
               "bias": float(work["err"].mean())}
    segments = {}
    for col in group_cols:
        if col in work.columns:
            segments[col] = segment_report(work, col, min_n=min_n, top=top_segments)
    curves = {}
    for feat in (feature_cols or []):
        if feat in work.columns and pd.api.types.is_numeric_dtype(work[feat]):
            curves[feat] = bias_curve(work, feat, bins=bins, min_n=min_n)
    worst = worst_cases(work, id_cols or [], k=k_worst,
                         extra_cols=[c for c in (feature_cols or []) if c in work.columns])
    return {"name": name, "overall": overall, "segments": segments, "bias_curves": curves,
            "worst_cases": worst}


def render_markdown(report: dict) -> str:
    """Turn a `build_report` payload into a markdown report, agent- and human-readable."""
    lines = [f"# Error explanation: {report['name']}", ""]
    o = report["overall"]
    lines += [f"**n = {o['n']:,} | RMSLE = {o['rmsle']:.4f} | mean bias = {o['bias']:+.4f}**",
              ""]
    for col, tbl in report["segments"].items():
        lines.append(f"## Segments: `{col}` (ranked by share of total squared error)")
        lines.append("")
        lines.append("| " + col + " | n | rmsle | bias | share_of_rows | share_of_error | lift |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for idx, row in tbl.iterrows():
            lines.append(f"| {idx} | {row['n']:,.0f} | {row['rmsle']:.4f} | {row['bias']:+.4f} "
                          f"| {row['share_of_rows']:.1%} | {row['share_of_error']:.1%} "
                          f"| {row['lift']:.2f}x |")
        lines.append("")
    for feat, curve in report["bias_curves"].items():
        mono = " -- **MONOTONIC BIAS, likely systematic miscalibration**" if \
            curve.attrs.get("monotonic_bias") else ""
        lines.append(f"## Bias curve: `{feat}`{mono}")
        lines.append("")
        lines.append("| bin | n | feature_mean | rmsle | bias |")
        lines.append("|---:|---:|---:|---:|---:|")
        for _, row in curve.iterrows():
            lines.append(f"| {row.name} | {row['n']:,.0f} | {row['feature_mean']:.3g} "
                          f"| {row['rmsle']:.4f} | {row['bias']:+.4f} |")
        lines.append("")
    wc = report["worst_cases"]
    if len(wc):
        lines.append(f"## Worst {len(wc)} individual rows")
        lines.append("")
        lines.append(_df_to_markdown_table(wc))
        lines.append("")
    return "\n".join(lines)


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Minimal pipe-table renderer -- avoids an optional `tabulate` dependency."""
    def fmt(v):
        return f"{v:+.4f}" if isinstance(v, (float, np.floating)) else str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def render_compare_markdown(diff: CompareResult) -> str:
    lines = [f"# Candidate diff by `{diff.name}`", "",
             f"**overall delta RMSLE = {diff.delta_rmsle:+.4f} (n={diff.n:,})**", ""]
    for title, tbl in (("Improved segments", diff.improved), ("Regressed segments", diff.regressed)):
        lines.append(f"## {title}")
        lines.append("")
        if len(tbl) == 0:
            lines.append("_none above the noise floor_\n")
            continue
        lines.append("| segment | n | rmsle_base | rmsle_cand | delta |")
        lines.append("|---|---:|---:|---:|---:|")
        for idx, row in tbl.iterrows():
            lines.append(f"| {idx} | {row['n']:,.0f} | {row['rmsle_base']:.4f} "
                          f"| {row['rmsle_cand']:.4f} | {row['delta']:+.4f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI: score one (or diff two) already-predicted files and write a report.

    Input files must be parquet or csv with, at minimum, the y/pred columns named on the
    command line. This does not train or predict anything -- see the module docstring for
    why that boundary is deliberate.
    """
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="parquet/csv of predictions to explain")
    p.add_argument("--compare-to", default=None, help="second file for a candidate A/B diff")
    p.add_argument("--id-col", default=None, help="row id column, required for --compare-to")
    p.add_argument("--y-col", default="y")
    p.add_argument("--pred-col", default="pred")
    p.add_argument("--group-cols", default="", help="comma-separated groupby columns")
    p.add_argument("--feature-cols", default="", help="comma-separated numeric feature columns")
    p.add_argument("--id-cols", default="", help="comma-separated columns to show in worst_cases")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=None, help="output .md path (default: reports/EXPLAIN_<name>.md)")
    args = p.parse_args()

    def _load(path: str) -> pd.DataFrame:
        return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    df = _load(args.input)
    name = args.name or Path(args.input).stem
    group_cols = [c for c in args.group_cols.split(",") if c]
    feature_cols = [c for c in args.feature_cols.split(",") if c]
    id_cols = [c for c in args.id_cols.split(",") if c]

    if args.compare_to:
        assert args.id_col, "--compare-to requires --id-col"
        cand = _load(args.compare_to)
        diff = compare(df, cand, id_col=args.id_col, y_col=args.y_col, pred_col=args.pred_col,
                        group_cols=group_cols or id_cols)
        out = Path(args.out or f"reports/EXPLAIN_DIFF_{name}.md")
        out.write_text(render_compare_markdown(diff), encoding="utf-8")
        print(f"wrote {out}")
        return

    report = build_report(df, y_col=args.y_col, pred_col=args.pred_col, group_cols=group_cols,
                           feature_cols=feature_cols, id_cols=id_cols, name=name)
    out = Path(args.out or f"reports/EXPLAIN_{name}.md")
    out.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
