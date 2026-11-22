"""
06_h1_test.py - FIN 611 fraud detection research project

WHAT THIS SCRIPT TESTS

  H1: choosing which condition to deploy by F1 costs more money than
      choosing it by cost.

  05_cost.py already produced the raw material. For each of the 15 outer
  folds it held out one fold, used the other 4 folds of the same repeat to
  pick (a) the F1-best condition and (b) the cost-best condition, then
  scored BOTH on the untouched held-out fold. The difference

      regret = cost of the F1-chosen condition
             - cost of the cost-chosen condition

  is one paired observation per fold, in dollars per 100,000 transactions.
  Positive means choosing on F1 was the more expensive mistake. Negative
  means F1 happened to pick the cheaper condition on that fold.

  This script takes those 15 numbers at each review fee and asks whether
  they sit above zero by more than chance.

WHY A WILCOXON SIGNED-RANK TEST AND NOT A t-TEST

  The 15 regrets are PAIRED: both conditions are scored on the same held-out
  fold, so fold-to-fold difficulty cancels out. That is what makes a paired
  test the right family.

  Within that family we use the signed-rank test rather than a paired t-test
  because the regret distribution is badly behaved for a t-test:
    * n = 15, far too small to lean on the central limit theorem;
    * it is heavily right-skewed - a handful of folds where F1 picked a
      genuinely bad condition dominate the mean;
    * it contains exact zeros (see below) and occasional large negatives.
  The signed-rank test only assumes the differences are symmetric about
  their median, and it ranks them rather than using their raw size, so one
  catastrophic fold cannot carry the result on its own.

  One-sided (alternative="greater") because H1 names a direction: we
  predicted F1-selection would be the costlier one, not merely different.
  Stating the direction in advance is what entitles us to the one-sided
  test and its extra power. The null is "median regret <= 0".

THE EXACT ZEROS - READ THIS BEFORE QUOTING ANY p-VALUE

  Some folds have regret of exactly 0.0. Those are not rounding: they are
  the folds where F1 and cost picked the SAME condition, so the two costs
  being differenced are literally the same number.

  scipy's default zero_method="wilcox" DISCARDS zero differences and shrinks
  n accordingly. So a fee with 2 such folds is tested on n = 13, not 15.
  That is the conventional choice and we keep it, but it is a conservative
  one: every discarded fold is a fold where the two selection criteria
  agreed, i.e. evidence AGAINST H1 being dropped from the sample. The
  script therefore prints n_eff alongside the p-value, and also reports the
  "pratt" variant, which keeps the zeros in the ranking, so you can see
  whether the choice moves the conclusion.

A LIMITATION YOU MUST STATE IN THE PAPER

  The signed-rank test assumes the 15 observations are independent. They
  are not. The outer design is RepeatedStratifiedKFold(5 x 3): within one
  repeat the 5 folds partition the data and are genuinely disjoint, but the
  3 repeats re-use the same rows in different arrangements, so every
  transaction contributes to 3 of the 15 held-out folds.

  The practical effect is that the true information content is closer to 5
  independent folds than 15, so these p-values are ANTI-CONSERVATIVE -
  smaller than they should be. Treat them as indicative, not exact, and say
  so. The honest headline is the effect size (median and mean regret in
  dollars), which the resampling does not bias; the p-value is a secondary
  check on sign consistency.

  Five fees means five tests, so the script also prints Holm-Bonferroni
  adjusted p-values for the family. Report the adjusted column if you want
  to claim significance "across fees" rather than at one chosen fee.

Run from anywhere:  python src/06_h1_test.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
COST_PATH = TABLE_DIR / "cost_comparison_all_conditions.json"
OUT_PATH = TABLE_DIR / "h1_wilcoxon.json"

ALPHA = 0.05


def load_fold_regrets(cost_json: dict) -> dict:
    """
    Pull the per-fold nested regret values out of 05_cost.py's output.

    Returns {fee (float): [15 regrets]}. The JSON stores fees as strings
    under "results" because JSON object keys are always strings, so we look
    each one up by str(fee) and fall back to a float match if 05_cost.py
    ever changes how it formats them.
    """
    fees = [float(f) for f in cost_json["fees"]]
    by_fee = {}

    for fee in fees:
        block = cost_json["results"].get(str(fee))
        if block is None:
            # Defensive: match on numeric value rather than string spelling.
            block = next(v for k, v in cost_json["results"].items()
                         if float(k) == fee)
        folds = block["nested_folds"]
        # Keep fold order stable and recorded, so a reader can trace any
        # single regret value back to the repeat/fold it came from.
        folds = sorted(folds, key=lambda r: (r["repeat"], r["fold"]))
        by_fee[fee] = {
            "regrets": [float(r["regret"]) for r in folds],
            "positions": [(int(r["repeat"]), int(r["fold"])) for r in folds],
            "same_pick": [bool(r["same_pick"]) for r in folds],
        }
    return by_fee


def holm_bonferroni(pvalues: list) -> list:
    """
    Holm-Bonferroni adjusted p-values, in the original input order.

    Plain Bonferroni multiplies every p by the number of tests, which is
    needlessly harsh. Holm sorts the p-values, multiplies the smallest by m,
    the next by m-1, and so on, then enforces that the adjusted sequence
    never decreases. It controls the family-wise error rate just as
    Bonferroni does but rejects more often.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        scaled = (m - rank) * pvalues[idx]
        running = max(running, scaled)          # keep it monotone
        adjusted[idx] = min(1.0, running)       # a p-value cannot exceed 1
    return adjusted


def test_one_fee(regrets: list) -> dict:
    """
    One-sided Wilcoxon signed-rank test of H0: median regret <= 0.

    scipy returns, for alternative="greater", W = the sum of the ranks of
    the POSITIVE differences. Bigger W means the positive regrets occupy
    more of the large ranks, which is what H1 predicts.
    """
    x = np.asarray(regrets, dtype=float)
    n = int(x.size)
    n_zero = int((x == 0).sum())
    n_eff = n - n_zero          # what zero_method="wilcox" actually tests on

    # The test needs at least one non-zero difference to rank.
    if n_eff == 0:
        return {"n": n, "n_zero": n_zero, "n_eff": 0,
                "median": float(np.median(x)), "mean": float(x.mean()),
                "sd": float(x.std(ddof=1)) if n > 1 else float("nan"),
                "n_negative": 0, "n_positive": 0,
                "statistic": float("nan"), "pvalue": float("nan"),
                "statistic_pratt": float("nan"), "pvalue_pratt": float("nan"),
                "supported": False}

    main = wilcoxon(x, alternative="greater")               # drops zeros
    pratt = wilcoxon(x, alternative="greater", zero_method="pratt")

    return {
        "n": n,
        "n_zero": n_zero,
        "n_eff": n_eff,
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "n_negative": int((x < 0).sum()),
        "n_positive": int((x > 0).sum()),
        "statistic": float(main.statistic),
        "pvalue": float(main.pvalue),
        "statistic_pratt": float(pratt.statistic),
        "pvalue_pratt": float(pratt.pvalue),
        "supported": bool(main.pvalue < ALPHA),
    }


def main() -> None:
    if not COST_PATH.exists():
        raise SystemExit(
            f"missing {COST_PATH.relative_to(PROJECT_ROOT)} - "
            f"run src/05_cost.py first")

    cost_json = json.loads(COST_PATH.read_text())
    by_fee = load_fold_regrets(cost_json)
    fees = sorted(by_fee)
    basis = cost_json.get("cost_basis", 100_000)

    print("=" * 100)
    print("H1: CHOOSING A CONDITION BY F1 COSTS MORE THAN CHOOSING BY COST")
    print("=" * 100)
    print(f"  Wilcoxon signed-rank, one-sided (alternative='greater').")
    print(f"  H0: median per-fold regret <= 0      H1: median > 0")
    print(f"  One observation per held-out outer fold; dollars per "
          f"{basis:,} transactions.")
    print(f"  Condition and threshold both chosen on the other folds of the "
          f"same repeat.")

    rows = {}
    for fee in fees:
        rows[fee] = test_one_fee(by_fee[fee]["regrets"])

    # Holm correction across the five fees, computed once all p's are in.
    adj = holm_bonferroni([rows[f]["pvalue"] for f in fees])
    for fee, p_adj in zip(fees, adj):
        rows[fee]["pvalue_holm"] = float(p_adj)
        rows[fee]["supported_holm"] = bool(p_adj < ALPHA)

    print("\n" + "-" * 100)
    print(f"{'fee':>8} {'median':>10} {'mean':>10} {'W':>8} {'p (1-sided)':>13} "
          f"{'neg':>6} {'n_eff':>6} {'H1 at 5%':>10}")
    print("-" * 100)
    for fee in fees:
        r = rows[fee]
        verdict = "SUPPORTED" if r["supported"] else "no"
        print(f"${fee:>7.2f} ${r['median']:>9,.0f} ${r['mean']:>9,.0f} "
              f"{r['statistic']:>8.1f} {r['pvalue']:>13.4f} "
              f"{r['n_negative']:>4}/{r['n']:<2} {r['n_eff']:>6} "
              f"{verdict:>10}")
    print("-" * 100)

    # ---- the caveats that decide how this gets written up ----------------
    print("\n  n_eff < n because zero_method='wilcox' discards folds whose "
          "regret is")
    print("  exactly zero - the folds where F1 and cost picked the SAME "
          "condition.")
    print("  Those are cases AGAINST H1, so dropping them is conservative. "
          "The")
    print("  'pratt' variant keeps them in the ranking:")
    print(f"\n{'fee':>8} {'p (wilcox)':>12} {'p (pratt)':>12} "
          f"{'p (Holm adj.)':>15} {'zeros':>7}")
    print("-" * 100)
    for fee in fees:
        r = rows[fee]
        print(f"${fee:>7.2f} {r['pvalue']:>12.4f} {r['pvalue_pratt']:>12.4f} "
              f"{r['pvalue_holm']:>15.4f} {r['n_zero']:>7}")
    print("-" * 100)

    n_sig = sum(r["supported"] for r in rows.values())
    n_sig_holm = sum(r["supported_holm"] for r in rows.values())
    medians = np.array([rows[f]["median"] for f in fees])
    print(f"\n  H1 supported at the 5% level at {n_sig} of {len(fees)} fees "
          f"({n_sig_holm} of {len(fees)} after Holm correction for the "
          f"{len(fees)} tests).")
    print(f"  Median regret runs ${medians.min():,.0f} to "
          f"${medians.max():,.0f} per {basis:,}; the sign is positive at "
          f"{int((medians > 0).sum())} of {len(fees)} fees.")
    print(f"\n  INDEPENDENCE CAVEAT: the {rows[fees[0]]['n']} folds come from "
          f"RepeatedStratifiedKFold(5 x 3),")
    print(f"  so each transaction appears in 3 held-out folds. The effective "
          f"information")
    print(f"  is nearer 5 independent folds than "
          f"{rows[fees[0]]['n']}, which makes these p-values")
    print(f"  ANTI-CONSERVATIVE. Lead with the dollar effect size; treat the "
          f"p-value as")
    print(f"  a check on sign consistency, and say this in the paper.")

    # ---- per-fold detail, so every number in the table is traceable ------
    # This block is wider than the 100 columns used above because it carries
    # one column per fold; the rule is sized to the row so it still lines up.
    positions = by_fee[fees[0]]["positions"]
    cell, gap = 6, 2
    width = 10 + len(positions) * cell + (len(positions) - 1) * gap
    print("\n" + "=" * width)
    print(f"PER-FOLD REGRET (dollars per {basis:,}; 'same' = both criteria "
          f"picked one condition)")
    print("=" * width)
    header = (" " * gap).join(f"{f'r{r}f{f}':>{cell}}" for r, f in positions)
    print(f"{'fee':>8}{' ' * gap}{header}")
    print("-" * width)
    for fee in fees:
        cells = []
        for val, same in zip(by_fee[fee]["regrets"], by_fee[fee]["same_pick"]):
            cells.append(f"{'same':>{cell}}" if same else f"{val:>{cell},.0f}")
        print(f"${fee:>7.2f}{' ' * gap}" + (" " * gap).join(cells))
    print("-" * width)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "hypothesis": "H1: selecting a condition by F1 costs more than "
                      "selecting by cost",
        "test": "Wilcoxon signed-rank, one-sided",
        "alternative": "greater",
        "null": "median per-fold nested regret <= 0",
        "alpha": ALPHA,
        "zero_method": "wilcox (zero differences discarded)",
        "cost_basis": basis,
        "units": f"dollars per {basis:,} transactions",
        "source": COST_PATH.name,
        "threshold_selection": cost_json.get("threshold_selection"),
        "conditions": cost_json.get("conditions"),
        "n_conditions": len(cost_json.get("conditions", [])),
        "multiple_comparison_correction": "Holm-Bonferroni across the "
                                          f"{len(fees)} fees",
        "independence_caveat": "The outer folds come from "
                               "RepeatedStratifiedKFold(5 x 3); each row "
                               "appears in 3 of the 15 held-out folds, so "
                               "the folds are not independent and the "
                               "p-values are anti-conservative.",
        "fees": fees,
        "results": {
            str(fee): {
                **rows[fee],
                "fold_positions": [f"r{r}f{f}"
                                   for r, f in by_fee[fee]["positions"]],
                "fold_regrets": by_fee[fee]["regrets"],
                "fold_same_pick": by_fee[fee]["same_pick"],
            }
            for fee in fees
        },
        "summary": {
            "n_fees": len(fees),
            "n_supported": int(n_sig),
            "n_supported_holm": int(n_sig_holm),
            "median_regret_min": float(medians.min()),
            "median_regret_max": float(medians.max()),
        },
    }, indent=2, default=float))
    print(f"\n  wrote {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
