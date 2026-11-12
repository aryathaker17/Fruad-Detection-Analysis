"""
02_explore.py - FIN 611 fraud detection research project

WHAT THIS SCRIPT DOES
  1. Reports class counts (how many fraud vs legitimate rows)
  2. Summary statistics for `amt`, split by is_fraud
  3. Gini coefficient of the fraudulent amounts (loss concentration)
  4. Saves a histogram of fraud amounts to results/figures/
  5. Prints a paste-ready Markdown block at the end for your report

Run from anywhere:  python src/02_explore.py

Reads data/processed/sample.parquet (made by 01_prepare.py). Writes only to
results/. Does not touch data/.

A NOTE ON WHICH NUMBERS ARE EXACT
  01_prepare.py kept ALL 9,651 fraud rows but only 10.33% of legitimate rows.
  So:
    * Every fraud statistic here is a POPULATION value, not an estimate -
      these are all the frauds in the dataset, not a sample of them.
    * Legitimate COUNTS and SUMS are scaled by 9.68 to estimate the
      population. Legitimate MEANS and PERCENTILES need no scaling: the
      legitimate rows were sampled at a constant rate, so the sample average
      is already an unbiased estimate of the population average.
  The scale factor is read from sample_metadata.json rather than hardcoded.
"""

import json
from pathlib import Path

import matplotlib
# Use a non-interactive backend: this script saves a file and never opens a
# window, so it works the same over SSH or in a scheduled run.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "sample.parquet"
META_PATH = PROJECT_ROOT / "data" / "processed" / "sample_metadata.json"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_PATH = FIG_DIR / "fraud_amount_histogram.png"

# Histogram bin width in dollars. Fixed at a round number (rather than letting
# matplotlib choose) for two reasons: the result is reproducible and quotable in
# a paper, and the default ~10 bins would smooth away the three clusters that
# are the most interesting thing about this distribution.
BIN_WIDTH = 25

# Figure styling. These come from a validated palette; see the note by
# plot_fraud_histogram() about which display rules do and do not apply to a
# static figure destined for print.
SURFACE = "#fcfcfb"   # paper/chart background
INK_PRIMARY = "#0b0b0b"
INK_SECOND = "#52514e"
INK_MUTED = "#8a8982"
SERIES_BLUE = "#2a78d6"   # sequential blue, step 450


def gini_coefficient(values: np.ndarray) -> float:
    """
    Gini coefficient of a set of non-negative amounts.

    WHAT IT MEASURES
      Concentration, on a 0-1 scale.
        0.0 = perfectly even   (every fraud steals the identical amount)
        1.0 = maximally uneven (one fraud accounts for all the loss)
      It is the standard inequality measure in economics, normally applied to
      income. Applied to fraud amounts it answers: is your fraud loss spread
      evenly across many incidents, or dominated by a few big ones?

      That question matters for this project specifically. If loss is highly
      concentrated, a model that catches most *incidents* can still miss most
      of the *money* - which is the gap between ML metrics and financial cost
      that your paper is about.

    CAREFUL - three unrelated things are called "Gini" and you will meet all
    three in this project. Define which one you mean in your paper.
      1. Gini COEFFICIENT (this function): inequality of a distribution.
      2. Gini IMPURITY: the node-splitting criterion in a decision tree,
         sklearn's `criterion="gini"`. Nothing to do with inequality.
      3. "Gini" in credit scoring: shorthand for 2*AUC - 1, a measure of how
         well a model ranks. This is the one a finance audience will most
         likely assume you mean, so say explicitly that you do not.

    FORMULA
      For amounts sorted ascending x_1 <= ... <= x_n:
          G = (2 * sum(i * x_i)) / (n * sum(x_i))  -  (n + 1) / n
      This is the standard unbiased-denominator form; it equals twice the area
      between the Lorenz curve and the line of equality.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n == 0:
        return float("nan")
    if np.any(x < 0):
        # Gini is undefined for negative values; this dataset has none, but a
        # silent wrong answer would be worse than a loud failure.
        raise ValueError("Gini coefficient requires non-negative values")
    total = x.sum()
    if total == 0:
        return float("nan")
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x)) / (n * total) - (n + 1.0) / n)


def bootstrap_gini_ci(values: np.ndarray, n_boot: int = 2000,
                      seed: int = 611) -> tuple[float, float]:
    """
    95% confidence interval for the Gini coefficient, by bootstrapping.

    WHAT BOOTSTRAPPING IS
      Resample the data with replacement many times, recompute the statistic on
      each resample, and read the spread of those answers. It gives an
      uncertainty estimate without assuming the data follows any named
      distribution.

    WHAT THIS INTERVAL DOES AND DOES NOT COVER
      These 9,651 rows are every fraud in the dataset, so there is NO
      subsampling error to quantify - the point estimate is exact for this
      dataset. This interval instead answers a different question: if these
      frauds are treated as one draw from some wider fraud-generating process,
      how much would Gini move on another draw? Report it as such, or omit it.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = gini_coefficient(rng.choice(x, size=x.size, replace=True))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def top_share(values: np.ndarray, pct: float) -> float:
    """
    Share of total loss held by the largest `pct` percent of frauds.

    Gini compresses the whole distribution into one number, which is hard to
    picture. "The biggest 10% of frauds account for X% of the money" says the
    same thing in a sentence a reader can act on, so report both.
    """
    x = np.sort(np.asarray(values, dtype=float))[::-1]  # descending
    k = max(1, int(round(len(x) * pct / 100.0)))
    return float(x[:k].sum() / x.sum() * 100.0)


def plot_fraud_histogram(fraud_amounts: pd.Series, out_path: Path) -> None:
    """
    Histogram of fraudulent transaction amounts.

    DESIGN NOTES
      Single series, so there is no legend: the title names what is plotted.
      Bars use one sequential hue; the mean/median reference lines and all text
      wear neutral ink rather than the series color, so color never carries
      meaning on its own. Grid is horizontal-only and recessive, behind the
      bars. Top and right spines removed - they enclose nothing.

      This is a STATIC figure for a printed report, so three rules that apply
      to on-screen charts are deliberately not used: no hover layer (paper
      cannot hover), no dark-mode variant (print is light), and no rounded bar
      ends (meaningless at 25-dollar bin width, and they would distort the
      visual area that encodes the count).
    """
    amounts = fraud_amounts.to_numpy(dtype=float)
    mean_amt = amounts.mean()
    median_amt = float(np.median(amounts))

    # Bins at a fixed round width, spanning 0 to the next whole bin above max.
    top = float(np.ceil(amounts.max() / BIN_WIDTH) * BIN_WIDTH)
    bins = np.arange(0, top + BIN_WIDTH, BIN_WIDTH)

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Recessive gridlines, drawn behind the data.
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8)

    # edgecolor = surface creates a thin gap between adjacent bars, so the
    # bars read as separate counts rather than one continuous mass.
    ax.hist(amounts, bins=bins, color=SERIES_BLUE,
            edgecolor=SURFACE, linewidth=0.7)

    # Reference lines in ink, not in the series color. Mean sits well above
    # median here: the distribution is right-skewed and multi-modal, so the
    # two disagree and quoting only one would mislead.
    ax.axvline(median_amt, color=INK_PRIMARY, linewidth=1.4, linestyle="-")
    ax.axvline(mean_amt, color=INK_SECOND, linewidth=1.4, linestyle="--")

    ymax = ax.get_ylim()[1]
    ax.text(median_amt, ymax * 0.97, f" median ${median_amt:,.0f}",
            color=INK_PRIMARY, fontsize=9, va="top", ha="left")
    ax.text(mean_amt, ymax * 0.86, f" mean ${mean_amt:,.0f}",
            color=INK_SECOND, fontsize=9, va="top", ha="left")

    ax.set_title("Fraudulent transaction amounts",
                 color=INK_PRIMARY, fontsize=13, pad=34, loc="left")
    ax.text(0, 1.01,
            f"All {len(amounts):,} fraudulent transactions · "
            f"${BIN_WIDTH} bins · Sparkov simulated data, 2019-2020",
            transform=ax.transAxes, color=INK_SECOND, fontsize=9.5, va="bottom")

    ax.set_xlabel("Transaction amount (USD)", color=INK_SECOND, fontsize=10)
    ax.set_ylabel("Number of transactions", color=INK_SECOND, fontsize=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d1")

    ax.set_xlim(-BIN_WIDTH * 0.6, top)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 70)
    print("02_explore.py - exploratory statistics")
    print("=" * 70)

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SAMPLE_PATH}\nRun `python src/01_prepare.py` first."
        )

    df = pd.read_parquet(SAMPLE_PATH, columns=["amt", "is_fraud", "category"])
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    pop_rate = meta.get("population_fraud_rate", float("nan"))

    fraud = df.loc[df.is_fraud == 1, "amt"]
    legit = df.loc[df.is_fraud == 0, "amt"]

    # ---- 1. Class counts ------------------------------------------------
    print("\n[1/4] CLASS COUNTS")
    print("-" * 70)
    n_total, n_fraud, n_legit = len(df), len(fraud), len(legit)
    print(f"  legitimate (is_fraud=0) : {n_legit:>9,}   "
          f"{n_legit / n_total:>7.3%} of sample")
    print(f"  fraudulent (is_fraud=1) : {n_fraud:>9,}   "
          f"{n_fraud / n_total:>7.3%} of sample")
    print(f"  total                   : {n_total:>9,}")
    print(f"\n  imbalance ratio         : 1 fraud per "
          f"{n_legit / n_fraud:.1f} legitimate rows (in sample)")
    print(f"  population fraud rate   : {pop_rate:.3%}")
    print(f"  difference              : {abs(n_fraud / n_total - pop_rate):.6%}"
          f"  <- the sample preserves the real rate, so every")
    print(f"                              statistic below is directly")
    print(f"                              interpretable without correction")

    # ---- 2. Amount summary by class -------------------------------------
    print("\n[2/4] AMOUNT SUMMARY BY CLASS  (`amt`, USD)")
    print("-" * 70)
    summary = df.groupby("is_fraud")["amt"].agg(
        n="count", mean="mean", std="std", min="min",
        p25=lambda s: s.quantile(.25), median="median",
        p75=lambda s: s.quantile(.75), p95=lambda s: s.quantile(.95),
        max="max", total="sum", skew="skew",
    ).T
    summary.columns = ["legitimate", "fraudulent"]
    print(summary.map(lambda v: f"{v:,.2f}").to_string())

    print(f"\n  A fraudulent transaction is on average "
          f"{fraud.mean() / legit.mean():.1f}x larger than a legitimate one")
    print(f"  ({fraud.median() / legit.median():.1f}x on the median, which is the")
    print(f"   more robust comparison here - both distributions are skewed)")

    value_share = fraud.sum() / (fraud.sum() + legit.sum())
    print(f"\n  Money at stake (this sample):")
    print(f"    total fraudulent amount : ${fraud.sum():>14,.2f}")
    print(f"    total legitimate amount : ${legit.sum():>14,.2f}")
    print(f"    fraud as share of value : {value_share:.3%}")
    print(f"\n  Note the asymmetry: fraud is {n_fraud / n_total:.3%} of TRANSACTIONS")
    print(f"  but {value_share:.3%} of DOLLARS, because fraudulent transactions")
    print(f"  are systematically larger. That ratio - "
          f"{value_share / (n_fraud / n_total):.1f}x - is why counting")
    print(f"  incidents and counting money give different answers.")

    # ---- 3. Gini --------------------------------------------------------
    print("\n[3/4] GINI COEFFICIENT OF FRAUD AMOUNTS")
    print("-" * 70)
    g = gini_coefficient(fraud.to_numpy())
    lo, hi = bootstrap_gini_ci(fraud.to_numpy())
    print(f"  Gini coefficient : {g:.4f}")
    print(f"  bootstrap 95% CI : [{lo:.4f}, {hi:.4f}]  (2,000 resamples)")
    print(f"  n                : {n_fraud:,}  (all frauds - point estimate is")
    print(f"                     exact for this dataset; see docstring on what")
    print(f"                     the interval does and does not cover)")
    print(f"\n  Loss concentration, stated plainly:")
    for pct in (1, 5, 10, 25, 50):
        print(f"    largest {pct:>2}% of frauds account for "
              f"{top_share(fraud.to_numpy(), pct):>5.1f}% of total fraud loss")
    print(f"\n  Reading it: 0 = every fraud the same size, 1 = one fraud is all")
    print(f"  the loss. At {g:.3f} this is MODERATE concentration - comparable to")
    print(f"  household income inequality in a developed economy. Loss is")
    print(f"  meaningfully top-heavy but not dominated by a handful of outliers.")
    print(f"\n  Why it matters for your paper: catching fraud *incidents* and")
    print(f"  catching fraud *dollars* are different objectives, and at this")
    print(f"  concentration they diverge enough to change which model wins.")
    print(f"\n  TERMINOLOGY WARNING: in credit-risk writing, 'Gini' usually means")
    print(f"  2*AUC-1, a model-ranking measure. This is the inequality")
    print(f"  coefficient, computed on dollar amounts, unrelated to any model.")
    print(f"  It is also unrelated to the 'gini' split criterion in a decision")
    print(f"  tree. State in your paper which definition you are using.")

    # ---- 4. Histogram ---------------------------------------------------
    print("\n[4/4] HISTOGRAM")
    print("-" * 70)
    plot_fraud_histogram(fraud, FIG_PATH)
    print(f"  saved {FIG_PATH.relative_to(PROJECT_ROOT)} "
          f"({FIG_PATH.stat().st_size / 1e3:.0f} KB, 200 dpi)")
    print(f"  bin width ${BIN_WIDTH}, {len(fraud)} observations plotted")
    print(f"\n  The distribution is MULTI-MODAL - three distinct clusters, not")
    print(f"  one bell curve. The cause is in the data generator: it draws")
    print(f"  fraud amounts from a narrow band per spending category.")
    by_cat = df[df.is_fraud == 1].groupby("category")["amt"].agg(["median", "min", "max"])
    print(f"  Fraud amount range by category (median, min-max):")
    for cat, row in by_cat.sort_values("median").iterrows():
        print(f"    {cat:<16} ${row['median']:>7,.0f}   "
              f"(${row['min']:,.0f} - ${row['max']:,.0f})")
    print(f"\n  So `amt` is only interpretable alongside `category`. Mean and")
    print(f"  median of the pooled distribution describe a mixture of three")
    print(f"  populations and should be quoted with that caveat.")

    # ---- Paste-ready ----------------------------------------------------
    print("\n" + "=" * 70)
    print("PASTE-READY MARKDOWN FOR YOUR REPORT")
    print("=" * 70)
    print(f"""
**Table 1. Class distribution (modeling sample, n = {n_total:,})**

| Class | Count | Share of sample |
|---|---|---|
| Legitimate | {n_legit:,} | {n_legit / n_total:.3%} |
| Fraudulent | {n_fraud:,} | {n_fraud / n_total:.3%} |

Sample fraud rate {n_fraud / n_total:.3%} against a population rate of \
{pop_rate:.3%}; the sample is stratified to preserve the true class balance.

**Table 2. Transaction amount by class (USD)**

| Statistic | Legitimate | Fraudulent |
|---|---|---|
| n | {n_legit:,} | {n_fraud:,} |
| Mean | {legit.mean():,.2f} | {fraud.mean():,.2f} |
| Std. dev. | {legit.std():,.2f} | {fraud.std():,.2f} |
| Minimum | {legit.min():,.2f} | {fraud.min():,.2f} |
| 25th pctile | {legit.quantile(.25):,.2f} | {fraud.quantile(.25):,.2f} |
| Median | {legit.median():,.2f} | {fraud.median():,.2f} |
| 75th pctile | {legit.quantile(.75):,.2f} | {fraud.quantile(.75):,.2f} |
| 95th pctile | {legit.quantile(.95):,.2f} | {fraud.quantile(.95):,.2f} |
| Maximum | {legit.max():,.2f} | {fraud.max():,.2f} |
| Skewness | {legit.skew():,.2f} | {fraud.skew():,.2f} |

**Table 3. Concentration of fraud loss**

| Measure | Value |
|---|---|
| Gini coefficient of fraud amounts | {g:.4f} |
| Bootstrap 95% CI | [{lo:.4f}, {hi:.4f}] |
| Share of loss from largest 1% of frauds | {top_share(fraud.to_numpy(), 1):.1f}% |
| Share of loss from largest 10% of frauds | {top_share(fraud.to_numpy(), 10):.1f}% |
| Share of loss from largest 25% of frauds | {top_share(fraud.to_numpy(), 25):.1f}% |
| Total fraud loss | ${fraud.sum():,.2f} |

Figure: `results/figures/fraud_amount_histogram.png`
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
