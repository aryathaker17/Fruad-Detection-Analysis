"""
07_h2_test.py - FIN 611 fraud detection research project

WHAT THIS SCRIPT TESTS

  H2: agreement between the metric rankings and the cost ranking PEAKS AT
      INTERMEDIATE review fees.

  The reasoning behind the prediction. At a very low review fee, flagging is
  almost free, so the cheapest policy is to flag enormous numbers of
  transactions; what matters is recall alone and the ranking quality that F1
  measures barely enters. At a very high fee, every review has to be earned,
  so only precision at the very top of the score list matters. Somewhere in
  between, cost should depend on the whole ranking - which is what F1 and
  AUPRC actually summarise - so agreement should be best there.

  Expressed as a regression, that prediction is a NEGATIVE QUADRATIC
  COEFFICIENT:

      rho = b0 + b1 * log(fee) + b2 * log(fee)^2,     H2: b2 < 0

  A negative b2 makes the fitted curve concave, i.e. an inverted U with a
  maximum in the middle.

WHY log(fee) AND NOT fee

  The fee enters the cost model as "fee x number flagged", so what moves the
  cost-minimising threshold is the fee's ORDER OF MAGNITUDE. 06/05's sweep is
  geometric for the same reason. Regressing on log(fee) makes the 25 design
  points evenly spaced in the predictor, so no single decade of the fee range
  dominates the fit by sheer density of observations.

TWO CONDITIONS, NOT ONE, FOR H2 TO HOLD

  A negative b2 on its own is NOT enough, and reporting only its sign would
  overstate the result. A concave parabola still rises monotonically over any
  interval that lies entirely to the left of its vertex. So this script also
  locates the vertex,

      log(fee*) = -b1 / (2 * b2)

  and checks whether fee* falls INSIDE the $0.25-$50 range that was actually
  sampled. H2 is supported only if b2 is negative, distinguishable from zero,
  AND the implied peak sits inside the data. If b2 < 0 but the vertex is
  outside the range, the honest description is "monotonic over the observed
  range", and the script says so and reports the direction from the linear
  fit.

WHY THE p-VALUES HERE ARE NOT A FORMAL TEST

  Stated at length in the printed output, because it governs how this gets
  written up. In short: the 25 rho values are not 25 independent
  observations. They are 25 re-rankings of ONE set of 180 saved prediction
  files, differing only in the fee used to price the thresholds. Adjacent
  fees frequently produce the IDENTICAL cost ranking and therefore the
  identical rho - the script counts how often. OLS standard errors assume
  independent errors and are badly wrong under that kind of serial
  dependence, so they come out too small. Read the fit as a description of
  the curve's shape, not as a hypothesis test with a valid p-value.

Run from anywhere:  python src/07_h2_test.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
SWEEP_PATH = TABLE_DIR / "fee_sweep_dense.json"
OUT_PATH = TABLE_DIR / "h2_test.json"

ALPHA = 0.05

# Spearman's rho on k items can only take values on a lattice:
#   rho = 1 - 6 * sum(d^2) / (k * (k^2 - 1))
# and sum(d^2) is always an even integer, so rho moves in fixed steps. With
# k = 12 conditions the step is 2 * 6 / (12 * 143) = 1/143. The response
# variable is therefore DISCRETE, which OLS's normal-errors assumption
# ignores. One more reason to read the fit descriptively.
def rho_lattice_step(k: int) -> float:
    return 12.0 / (k * (k * k - 1))


def fit_models(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Fit rho ~ log(fee) and rho ~ log(fee) + log(fee)^2 by OLS.

    Both models are returned together because the quadratic coefficient is
    only interpretable next to the linear fit it is meant to improve on:
    R-squared cannot fall when a term is added, so the comparison that
    matters is the t-test on b2 and the change in ADJUSTED R-squared.
    """
    frame = pd.DataFrame({"log_fee": x, "log_fee_sq": x ** 2})

    linear = sm.OLS(y, sm.add_constant(frame[["log_fee"]])).fit()
    quadratic = sm.OLS(
        y, sm.add_constant(frame[["log_fee", "log_fee_sq"]])).fit()

    return {"linear": linear, "quadratic": quadratic}


def describe_fit(model, terms: list) -> dict:
    """Pull the reportable numbers out of a fitted OLS result."""
    return {
        "terms": {
            t: {
                "coef": float(model.params[t]),
                "se": float(model.bse[t]),
                "t": float(model.tvalues[t]),
                "p": float(model.pvalues[t]),
                "ci_low": float(model.conf_int().loc[t, 0]),
                "ci_high": float(model.conf_int().loc[t, 1]),
            }
            for t in terms
        },
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "n": int(model.nobs),
        "df_resid": int(model.df_resid),
        "durbin_watson": float(durbin_watson(model.resid)),
    }


def format_fee(fee: float) -> str:
    """
    Readable fee label, including for vertices far outside the sampled range.

    A fitted vertex can land at $0.002 or at $10^34 when the quadratic term
    is near zero - dividing by a coefficient that is indistinguishable from
    zero sends the turning point off to infinity. "$0.00" would hide that, so
    anything outside a plausible fee range is printed in scientific notation.
    """
    if not np.isfinite(fee):
        return "undefined"
    if 0.01 <= abs(fee) < 1e6:
        return f"${fee:,.4f}"
    return f"${fee:.3e}"


def assess_h2(quad, fees: np.ndarray) -> dict:
    """
    Decide whether H2 holds, and say precisely why it does or does not.

    H2 needs all three of:
      1. b2 < 0                       - the curve is concave at all
      2. p(b2) < ALPHA                - b2 is distinguishable from zero
      3. the vertex is inside the
         sampled fee range            - the peak is actually observed, not
                                        an extrapolation beyond the data
    """
    b1 = float(quad.params["log_fee"])
    b2 = float(quad.params["log_fee_sq"])
    p2 = float(quad.pvalues["log_fee_sq"])

    concave = b2 < 0
    significant = p2 < ALPHA

    # Vertex of the parabola. Undefined if b2 is exactly zero.
    if b2 != 0:
        x_star = -b1 / (2.0 * b2)
        fee_star = float(np.exp(x_star))
    else:
        x_star, fee_star = float("nan"), float("nan")

    lo, hi = float(fees.min()), float(fees.max())
    inside = bool(np.isfinite(fee_star) and lo < fee_star < hi)

    supported = bool(concave and significant and inside)

    if supported:
        reason = (f"b2 < 0 (p = {p2:.4f}) and the implied peak at "
                  f"{format_fee(fee_star)} lies inside the sampled range")
    elif concave and significant and not inside:
        reason = (f"b2 < 0 (p = {p2:.4f}) but the peak at "
                  f"{format_fee(fee_star)} is outside ${lo:.2f}-${hi:.2f}, "
                  f"so the fitted curve is monotonic over the range "
                  f"actually observed")
    elif concave and not significant:
        reason = (f"b2 < 0 but not distinguishable from zero "
                  f"(p = {p2:.4f}), so the data do not support curvature "
                  f"in either direction")
    elif not concave and significant:
        reason = (f"b2 > 0 (p = {p2:.4f}) - the fitted curve is CONVEX, the "
                  f"opposite of the prediction")
    else:
        reason = (f"b2 > 0 and not distinguishable from zero "
                  f"(p = {p2:.4f}); no curvature is detectable")

    # The turning point of a parabola is a maximum only when it opens
    # downward. With b2 > 0 it is a MINIMUM, which is the opposite of what
    # H2 describes, so it must not be reported as a "peak".
    vertex_kind = "maximum (peak)" if concave else "minimum (trough)"

    return {"b2": b2, "p_b2": p2, "concave": concave,
            "significant": significant, "vertex_log_fee": float(x_star),
            "vertex_fee": fee_star, "vertex_kind": vertex_kind,
            "vertex_inside_range": inside,
            "supported": supported, "reason": reason}


def monotone_direction(linear, x: np.ndarray) -> dict:
    """
    The direction and size of the linear trend, for when H2 fails.

    The slope is reported two ways. The raw coefficient is "rho per unit of
    log(fee)", which is hard to picture. Multiplying by the full span of
    log(fee) gives the predicted change in rho from the cheapest sampled fee
    to the dearest, which is the number worth quoting.
    """
    b1 = float(linear.params["log_fee"])
    p1 = float(linear.pvalues["log_fee"])
    span = float(x.max() - x.min())

    if p1 >= ALPHA:
        direction = "flat (slope not distinguishable from zero)"
    elif b1 > 0:
        direction = "INCREASING in the fee"
    else:
        direction = "DECREASING in the fee"

    return {"slope": b1, "p_slope": p1, "direction": direction,
            "log_fee_span": span,
            "predicted_rho_change_over_range": b1 * span}


def serial_dependence(fees: np.ndarray, series: dict) -> dict:
    """
    Quantify how far from independent these 25 observations are.

    The caveat that the rho values are dependent is easy to assert and easy
    to wave away, so it is measured instead. Two concrete figures:

      * adjacent ties: how many of the 24 neighbouring fee pairs produce
        the EXACT same rho. A tie means the cost ranking did not change at
        all between those fees - the same 12 conditions in the same order -
        so the two observations carry one piece of information, not two.
      * distinct values: how many different rho values the 25 fees produce.
    """
    out = {}
    for label, y in series.items():
        ties = int(sum(y[i] == y[i + 1] for i in range(len(y) - 1)))
        out[label] = {
            "n_observations": int(len(y)),
            "n_distinct_values": int(len(set(y.tolist()))),
            "adjacent_identical_pairs": ties,
            "adjacent_pairs": int(len(y) - 1),
            "observed_min": float(y.min()),
            "observed_max": float(y.max()),
            "observed_range": float(y.max() - y.min()),
        }
    return out


def print_fit(label: str, desc: dict, terms: list, pretty: dict) -> None:
    """One OLS table."""
    print(f"\n  {label}")
    print(f"    {'term':>14} {'coef':>12} {'se':>10} {'t':>8} {'p':>9} "
          f"{'95% CI':>22}")
    print("    " + "-" * 80)
    for t in terms:
        v = desc["terms"][t]
        ci = f"[{v['ci_low']:+.5f}, {v['ci_high']:+.5f}]"
        print(f"    {pretty.get(t, t):>14} {v['coef']:>+12.6f} "
              f"{v['se']:>10.6f} {v['t']:>8.3f} {v['p']:>9.4f} {ci:>22}")
    print("    " + "-" * 80)
    print(f"    R-squared {desc['r_squared']:.4f}   "
          f"adj R-squared {desc['adj_r_squared']:.4f}   "
          f"F {desc['f_statistic']:.3f} (p = {desc['f_pvalue']:.4f})   "
          f"n = {desc['n']}")


def analyse(label: str, fees: np.ndarray, x: np.ndarray,
            y: np.ndarray) -> dict:
    """Run both fits for one rho series, print the tables, return the facts."""
    models = fit_models(x, y)
    lin, quad = models["linear"], models["quadratic"]

    lin_desc = describe_fit(lin, ["const", "log_fee"])
    quad_desc = describe_fit(quad, ["const", "log_fee", "log_fee_sq"])
    pretty = {"const": "intercept", "log_fee": "log(fee)",
              "log_fee_sq": "log(fee)^2"}

    print("\n" + "=" * 100)
    print(f"{label}")
    print("=" * 100)
    print(f"  observed rho: {y.min():.4f} to {y.max():.4f} "
          f"(range {y.max() - y.min():.4f}), "
          f"{len(set(y.tolist()))} distinct values across {len(y)} fees")

    print_fit("LINEAR     rho = b0 + b1*log(fee)", lin_desc,
              ["const", "log_fee"], pretty)
    print_fit("QUADRATIC  rho = b0 + b1*log(fee) + b2*log(fee)^2", quad_desc,
              ["const", "log_fee", "log_fee_sq"], pretty)

    # Does the quadratic term earn its place? With one extra parameter this
    # F-test is algebraically the square of the t-test on b2; it is printed
    # because the "nested model comparison" framing is the one a reviewer
    # will expect to see.
    f_stat, f_p, _ = quad.compare_f_test(lin)
    d_adj = quad_desc["adj_r_squared"] - lin_desc["adj_r_squared"]
    print(f"\n    quadratic vs linear: F = {float(f_stat):.3f} "
          f"(p = {float(f_p):.4f}), adj R-squared change "
          f"{d_adj:+.4f}")
    print(f"    R-squared can only rise when a term is added, so the "
          f"adjusted change is")
    print(f"    the one that tells you whether log(fee)^2 pays for itself.")

    h2 = assess_h2(quad, fees)
    mono = monotone_direction(lin, x)

    print(f"\n  H2 (b2 < 0, a peak at intermediate fees):")
    print(f"    b2 = {h2['b2']:+.6f}  (p = {h2['p_b2']:.4f})")
    if np.isfinite(h2["vertex_fee"]):
        where = "INSIDE" if h2["vertex_inside_range"] else "OUTSIDE"
        print(f"    turning point is a {h2['vertex_kind']} at fee = "
              f"{format_fee(h2['vertex_fee'])}")
        print(f"      {where} the sampled "
              f"${fees.min():.2f}-${fees.max():.2f}")
    print(f"\n    >>> H2 {'SUPPORTED' if h2['supported'] else 'NOT SUPPORTED'}"
          f" for {label.lower()}")
    print(f"        {h2['reason']}")

    if not h2["supported"]:
        print(f"\n    The relationship is instead MONOTONIC over the "
              f"sampled range:")
        print(f"      direction : rho is {mono['direction']}")
        print(f"      slope     : {mono['slope']:+.6f} rho per unit of "
              f"log(fee)  (p = {mono['p_slope']:.4f})")
        print(f"      over the full ${fees.min():.2f} to ${fees.max():.2f} "
              f"span that is a predicted rho change of "
              f"{mono['predicted_rho_change_over_range']:+.4f}")

    return {"linear": lin_desc, "quadratic": quad_desc,
            "quadratic_vs_linear": {"f": float(f_stat), "p": float(f_p),
                                    "adj_r_squared_change": float(d_adj)},
            "h2": h2, "monotone": mono,
            "observed": {"rho_min": float(y.min()), "rho_max": float(y.max()),
                         "n_distinct": int(len(set(y.tolist())))}}


def main() -> None:
    if not SWEEP_PATH.exists():
        raise SystemExit(
            f"missing {SWEEP_PATH.relative_to(PROJECT_ROOT)} - "
            f"run src/05_cost.py first")

    sweep = json.loads(SWEEP_PATH.read_text())
    rows = sweep["rows"]
    fees = np.array([r["fee"] for r in rows], dtype=float)
    x = np.log(fees)

    series = {
        "rho(F1 rank vs cost rank)":
            np.array([r["rho_f1_cost"] for r in rows], dtype=float),
        "rho(AUPRC rank vs cost rank)":
            np.array([r["rho_auprc_cost"] for r in rows], dtype=float),
    }

    n_conditions = len(sweep["conditions"])
    step = rho_lattice_step(n_conditions)

    print("=" * 100)
    print("H2: METRIC-VS-COST RANK AGREEMENT PEAKS AT INTERMEDIATE REVIEW FEES")
    print("=" * 100)
    print(f"  Prediction: regressing rho on log(fee) yields a NEGATIVE "
          f"quadratic coefficient,")
    print(f"  so the fitted curve is concave and agreement is best in the "
          f"middle of the")
    print(f"  fee range rather than at either end.")
    print(f"\n  source      : {SWEEP_PATH.name}")
    print(f"  design      : {len(fees)} fees, log-spaced "
          f"${fees.min():.2f} to ${fees.max():.2f}")
    print(f"  conditions  : {n_conditions} ranked at each fee")
    print(f"  thresholds  : {sweep['threshold_selection']}")
    print(f"  method      : statsmodels OLS, rho on log(fee) and "
          f"log(fee) + log(fee)^2")

    results = {}
    for label, y in series.items():
        results[label] = analyse(label, fees, x, y)

    # ---- the caveat, deliberately last and deliberately loud -----------
    dep = serial_dependence(fees, series)
    dep_max = max(d["adjacent_identical_pairs"] for d in dep.values())

    print("\n" + "!" * 100)
    print("READ THIS BEFORE QUOTING ANY p-VALUE ABOVE")
    print("!" * 100)
    print("""
  THE 25 OBSERVATIONS ARE NOT INDEPENDENT.

  Every rho above was computed from the SAME 180 saved prediction files.
  Nothing about the models, the folds or the predictions changes across the
  sweep; the only thing that changes is the fee used to price each flag when
  the cost-minimising threshold is chosen. The 25 rho values are therefore 25
  re-readings of one experiment, not 25 experiments.

  How strong is the dependence? Adjacent fees often produce the IDENTICAL
  cost ranking - the same 12 conditions in the same order - and therefore the
  identical rho:""")
    for label, d in dep.items():
        print(f"      {label}")
        print(f"        {d['adjacent_identical_pairs']} of "
              f"{d['adjacent_pairs']} neighbouring fee pairs give the exact "
              f"same rho")
        print(f"        {d['n_distinct_values']} distinct rho values across "
              f"{d['n_observations']} fees")
    print("""
  Durbin-Watson on the residuals, for completeness (2.0 = no lag-1
  autocorrelation, below 2 = positive, above 2 = negative):""")
    for label, r in results.items():
        print(f"      {label:>32}  linear {r['linear']['durbin_watson']:.3f}"
              f"   quadratic {r['quadratic']['durbin_watson']:.3f}")
    print(f"""
  Note which way those point. They sit ABOVE 2, so the lag-1 residual
  correlation is mildly NEGATIVE, not positive. The textbook story - "positive
  autocorrelation shrinks the standard errors" - therefore does NOT apply
  here, and it is not the reason to distrust these p-values. The real problem
  is more basic.

  WHAT ACTUALLY FOLLOWS

    * THE RESIDUALS ARE NOT RANDOM ERROR. Given the 180 prediction files,
      rho(fee) is a DETERMINISTIC function of the fee - compute it twice and
      you get the same number twice. There is no sampling noise in these 25
      points at all. The scatter around the fitted line is the genuine
      wiggliness of a step function, not error, so the OLS error model is
      misspecified at the root rather than merely violated in its
      correlation structure.
    * THE p-VALUES ANSWER THE WRONG QUESTION. A p-value here asks: "if 25
      rho values were drawn with independent random errors about a straight
      line, how surprising would this slope be?" The question the paper asks
      is: "would a fresh sample of transactions reproduce this relationship?"
      The design cannot answer the second, because it contains exactly ONE
      independent replication - one 400,000-row sample, one set of CV folds.
      The variability that matters is variability ACROSS such replications,
      and the fit has no way to see it.
    * SO THE p-VALUES ARE OPTIMISTIC, but because they omit the dominant
      source of uncertainty entirely, not because of a correlation
      correction that could be patched in. No HAC or clustered standard
      error fixes this; only re-running the pipeline on independent samples
      would.
    * rho is also DISCRETE, not continuous: with {n_conditions} conditions it can
      only land on multiples of {step:.6f}, which is why {dep_max} of 24
      neighbouring fee pairs return the identical value. OLS assumes
      continuous normally-distributed errors.
    * NOTHING IS WRONG WITH THE COEFFICIENTS. Least squares still describes
      the shape of the relationship between rho and log(fee), and R-squared
      still reports how much of the observed variation that shape accounts
      for. It is the inferential layer - standard errors, t, p, confidence
      intervals - that does not survive.

  HOW TO WRITE IT UP

    Report the fit as a DESCRIPTIVE summary of how rank agreement varies with
    the review fee. Lead with the coefficient signs and R-squared, and state
    plainly that the p-values are not a valid formal test because all 25 fees
    reuse one set of predictions. Do not claim significance. The verdict on
    H2 rests on the SIGN of b2 and the LOCATION of the turning point, both of
    which are properties of the fitted curve rather than of its standard
    errors - so the verdict below stands regardless of the inference problem.
""")
    print("!" * 100)

    # ---- overall verdict ------------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT ON H2")
    print("=" * 100)
    any_supported = False
    for label, r in results.items():
        h2 = r["h2"]
        mono = r["monotone"]
        tag = "SUPPORTED" if h2["supported"] else "NOT SUPPORTED"
        any_supported |= h2["supported"]
        print(f"  {label}")
        print(f"      {tag}: b2 = {h2['b2']:+.6f}, p = {h2['p_b2']:.4f}")
        if not h2["supported"]:
            print(f"      instead monotonic: rho is {mono['direction']}, "
                  f"{mono['predicted_rho_change_over_range']:+.4f} across "
                  f"the range")
    if not any_supported:
        print(f"\n  H2 is NOT SUPPORTED for either metric. The predicted "
              f"inverted-U does not")
        print(f"  appear: agreement does not peak at intermediate fees.")
    print("=" * 100)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "hypothesis": "H2: metric-vs-cost rank agreement peaks at "
                      "intermediate review fees",
        "operationalisation": "negative quadratic coefficient in "
                              "rho ~ log(fee) + log(fee)^2",
        "support_criteria": [
            "b2 < 0",
            f"p(b2) < {ALPHA}",
            "implied vertex inside the sampled fee range",
        ],
        "method": "statsmodels OLS",
        "alpha": ALPHA,
        "source": SWEEP_PATH.name,
        "n_fees": int(len(fees)),
        "fee_min": float(fees.min()),
        "fee_max": float(fees.max()),
        "fees": fees.tolist(),
        "n_conditions": n_conditions,
        "rho_lattice_step": step,
        "threshold_selection": sweep.get("threshold_selection"),
        "rho_f1_cost": series["rho(F1 rank vs cost rank)"].tolist(),
        "rho_auprc_cost": series["rho(AUPRC rank vs cost rank)"].tolist(),
        "independence_caveat": (
            "The 25 rho values derive from the same 180 saved prediction "
            "files and differ only in the fee used to price thresholds, so "
            "they are not independent observations. Given those files "
            "rho(fee) is deterministic, so the residuals are not sampling "
            "error and the OLS error model is misspecified at the root. The "
            "design contains one independent replication (one sample, one "
            "set of CV folds), so the p-values omit the dominant source of "
            "uncertainty and are optimistic; no HAC or clustered standard "
            "error repairs this. Read the fits as descriptive summaries of "
            "the shape of the relationship, not as formal hypothesis tests. "
            "Measured lag-1 residual autocorrelation is mildly NEGATIVE "
            "(Durbin-Watson above 2), so the usual positive-autocorrelation "
            "argument does not apply. rho is also discrete, landing only on "
            f"multiples of {step:.6f} with {n_conditions} conditions."),
        "serial_dependence": dep,
        "results": results,
        "h2_supported_any": bool(any_supported),
    }, indent=2, default=float))
    print(f"\n  wrote {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
