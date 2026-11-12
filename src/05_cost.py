"""
05_cost.py - FIN 611 fraud detection research project

THE COST MODEL
  Every transaction falls into one of four outcomes. This is what each costs:

                        |  flagged for review   |  let through
    --------------------|-----------------------|------------------
    actually fraud      |  review fee           |  the full amount
    actually legitimate |  review fee           |  nothing

  In words: reviewing anything costs the fee, whether or not it turns out to
  be fraud. Missing a fraud costs the money that was stolen. Correctly letting
  a legitimate transaction through is free.

  ASSUMPTIONS BUILT INTO THAT TABLE - state these in your paper:
    * A flagged fraud is fully recovered, so it costs only the review fee.
      Real recovery is partial and sometimes fails.
    * A flagged legitimate transaction costs only the fee - no customer
      annoyance, no abandoned purchase, no churn. In practice a false decline
      is often far more expensive than the review itself.
    * The fee is constant per review. Real queues have tiers.

UNITS
  Every dollar figure is reported PER 100,000 TRANSACTIONS. Raw fold totals
  depend on how big the fold happened to be, which makes them impossible to
  compare across folds, across models, or against any published figure.
  Normalising fixes the denominator once so every number in this script is
  directly comparable.

  No class reweighting appears anywhere. 01_prepare.py draws a sample that
  preserves the true fraud rate, so the sample already is a miniature of the
  population and costs scale linearly with transaction count.

WHY THIS SCRIPT EXISTS
  A threshold of 0.5 is an arbitrary inheritance from balanced classification.
  Nothing about fraud makes it the right cutoff. This script finds the cutoff
  that actually minimizes money lost, and shows what it costs to use 0.5.

Run from anywhere:  python src/05_cost.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT_ROOT / "results" / "predictions"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

# Cost of one manual review, in dollars. There is no single correct value -
# published estimates for card-fraud review run from roughly $1 to $25
# depending on how much analyst time and tooling is loaded in. $3 is a
# mid-range figure for an automated-first queue. Because the optimal threshold
# depends directly on this number, the script also prints a sensitivity table.
# Conditions produced by 04_grid.py are named <model>__<strategy>_r#_f#.npz.
CONDITION_GLOB = "*__*_r*_f*.npz"

# Every fee at which the whole comparison is repeated. The optimal threshold,
# and therefore which condition is cheapest, depends on this number - so the
# comparison is only meaningful when repeated across a plausible range.
FEES = [0.50, 1.00, 3.00, 5.00, 25.00]

# The fee used for the single detailed leakage check.
REFERENCE_FEE = 3.00

DEFAULT_THRESHOLD = 0.5

# All costs are expressed per this many transactions.
COST_BASIS = 100_000


def total_cost(probabilities, labels, amounts, threshold, review_fee):
    """
    Total dollar cost of operating the model at a given threshold.

    Parameters
    ----------
    probabilities : predicted P(fraud) for each transaction
    labels        : true is_fraud, 0 or 1
    amounts       : transaction amount in dollars
    threshold     : flag a transaction when probability >= threshold
    review_fee    : dollar cost of reviewing one flagged transaction

    Returns
    -------
    float : missed-fraud losses + review fees, as a raw total for the rows
            passed in. Use per_basis() to express it per 100,000 transactions.

    The formula:
        cost = SUM(amount of every fraud NOT flagged)
             + review_fee * COUNT(everything flagged)
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    amounts = np.asarray(amounts, dtype=float)

    flagged = probabilities >= threshold

    # A fraud that was not flagged: the money is gone.
    missed_fraud_loss = float(np.sum(amounts[(labels == 1) & ~flagged]))
    # Everything flagged goes to a human, fraud or not.
    review_cost = float(review_fee * np.count_nonzero(flagged))

    return missed_fraud_loss + review_cost


def per_basis(cost, n_rows):
    """Express a raw cost per COST_BASIS transactions."""
    return cost / n_rows * COST_BASIS


def cost_minimising_threshold(probabilities, labels, amounts, review_fee):
    """
    The exact threshold that minimizes total_cost, and that minimum.

    HOW IT WORKS - no grid search, no approximation.
      Sort every transaction by predicted probability, highest first. Any
      threshold is then equivalent to "flag the top k". Walking k from 0 to n
      and tracking two running totals gives the exact cost at every possible
      cutoff in one pass:

        cost(k) = (all fraud money) - (fraud money inside the top k)
                + fee * k

      Each extra flag adds one review fee and recovers that row's fraud amount
      if it was fraud. The best k is wherever the running total bottoms out.
      A probability grid would only sample this curve; sorting evaluates it
      everywhere a change can occur.
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    amounts = np.asarray(amounts, dtype=float)

    order = np.argsort(-probabilities, kind="mergesort")  # stable, descending
    p_sorted = probabilities[order]
    fraud_value = amounts[order] * (labels[order] == 1)

    total_fraud_value = fraud_value.sum()
    # [0] prepended so index k means "flagged the top k rows".
    recovered = np.concatenate([[0.0], np.cumsum(fraud_value)])
    n_flagged = np.arange(len(p_sorted) + 1)

    costs = (total_fraud_value - recovered) + review_fee * n_flagged
    k = int(np.argmin(costs))

    if k == 0:
        # Flag nothing. Any threshold above the highest score does this.
        return float(np.nextafter(p_sorted[0], np.inf)), float(costs[0])

    threshold = float(p_sorted[k - 1])
    # Ties: if the k-th and (k+1)-th rows share a probability, a >= cutoff
    # flags both. The difference is negligible but the reported threshold
    # should still reproduce the reported cost, so recompute rather than
    # trusting costs[k].
    return threshold, total_cost(probabilities, labels, amounts,
                                 threshold, review_fee)


def pooled_threshold(selection_folds, review_fee):
    """
    Cost-minimising threshold chosen on a POOL of folds.

    Concatenates the selection folds into one set of predictions and finds the
    single threshold that minimizes cost across all of them. Cost is additive
    over rows, so pooling is the right way to combine folds: it weights each
    fold by its size and finds the cutoff that is best on average, which is
    what you would actually deploy.
    """
    y_prob = np.concatenate([f["y_prob"] for f in selection_folds])
    y_true = np.concatenate([f["y_true"] for f in selection_folds])
    amt = np.concatenate([f["amt"] for f in selection_folds])
    threshold, _ = cost_minimising_threshold(y_prob, y_true, amt, review_fee)
    return threshold


def evaluate_at(fold, threshold, review_fee):
    """Score one fold at a threshold chosen elsewhere."""
    n = len(fold["y_true"])
    pred = (fold["y_prob"] >= threshold).astype(int)
    cost = per_basis(total_cost(fold["y_prob"], fold["y_true"], fold["amt"],
                                threshold, review_fee), n)
    return {
        "threshold": threshold,
        "cost": cost,
        "f1": f1_score(fold["y_true"], pred, zero_division=0),
        "flag_rate": float(pred.mean()),
    }


def load_folds(pattern: str):
    """Load every saved fold matching a filename pattern."""
    files = sorted(PRED_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No prediction files matching {pattern!r} in {PRED_DIR}\n"
            "Run `python src/03_baseline.py` or `python src/04_grid.py` first."
        )
    folds = []
    for f in files:
        d = np.load(f)
        folds.append({
            "name": f.stem, "repeat": int(d["repeat"]), "fold": int(d["fold"]),
            "y_true": d["y_true"], "y_prob": d["y_prob"], "amt": d["amt"],
        })
    return folds


def discover_conditions() -> dict:
    """
    Group saved prediction files by condition.

    Returns {condition_name: [fold dicts]}, where the condition is the
    <model>__<strategy> part of the filename written by 04_grid.py.
    """
    files = sorted(PRED_DIR.glob(CONDITION_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No condition files matching {CONDITION_GLOB!r} in {PRED_DIR}\n"
            "Run `python src/04_grid.py` first.")

    conditions = {}
    for f in files:
        # "<model>__<strategy>_r<repeat>_f<fold>" -> "<model>__<strategy>"
        condition = f.stem.rsplit("_r", 1)[0]
        d = np.load(f)
        conditions.setdefault(condition, []).append({
            "name": f.stem, "repeat": int(d["repeat"]), "fold": int(d["fold"]),
            "y_true": d["y_true"], "y_prob": d["y_prob"], "amt": d["amt"],
        })
    for folds in conditions.values():
        folds.sort(key=lambda x: (x["repeat"], x["fold"]))
    return conditions


def score_condition(folds, fee):
    """
    Evaluate one condition at one review fee.

    The cost-minimising threshold is chosen on the OTHER FOLDS OF THE SAME
    REPEAT and scored on the held-out fold. Within a repeat the 5 folds
    partition the data, so the selection folds share no rows with the fold
    being scored - the threshold never sees the labels it is graded against.
    """
    rows = []
    for fd in folds:
        n = len(fd["y_true"])

        pred_05 = (fd["y_prob"] >= DEFAULT_THRESHOLD).astype(int)
        f1_05 = f1_score(fd["y_true"], pred_05, zero_division=0)
        cost_05 = per_basis(total_cost(fd["y_prob"], fd["y_true"], fd["amt"],
                                       DEFAULT_THRESHOLD, fee), n)
        auprc = average_precision_score(fd["y_true"], fd["y_prob"])

        others = [f for f in folds
                  if f["repeat"] == fd["repeat"] and f["name"] != fd["name"]]
        thr = pooled_threshold(others, fee)
        held = evaluate_at(fd, thr, fee)

        # The leaky comparator, kept only to quantify selection bias.
        thr_in = pooled_threshold([fd], fee)
        in_sample = evaluate_at(fd, thr_in, fee)

        rows.append({
            "f1_05": f1_05, "cost_05": cost_05, "auprc": auprc,
            "thr": thr, "f1_opt": held["f1"], "cost_opt": held["cost"],
            "flag_opt": held["flag_rate"],
            "cost_opt_leaky": in_sample["cost"],
        })

    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {
        "cost_opt_sd": float(np.std([r["cost_opt"] for r in rows], ddof=1)),
        "f1_05_sd": float(np.std([r["f1_05"] for r in rows], ddof=1)),
    }


def pooled_cost(folds_subset, threshold, review_fee):
    """Cost per COST_BASIS over a pool of folds, all scored at one threshold."""
    y_prob = np.concatenate([f["y_prob"] for f in folds_subset])
    y_true = np.concatenate([f["y_true"] for f in folds_subset])
    amt = np.concatenate([f["amt"] for f in folds_subset])
    return per_basis(
        total_cost(y_prob, y_true, amt, threshold, review_fee), len(y_true))


def pooled_f1(folds_subset, threshold=DEFAULT_THRESHOLD):
    """F1 over a pool of folds at a fixed threshold."""
    y_prob = np.concatenate([f["y_prob"] for f in folds_subset])
    y_true = np.concatenate([f["y_true"] for f in folds_subset])
    return f1_score(y_true, (y_prob >= threshold).astype(int),
                    zero_division=0)


def nested_regret(conditions, names, fee):
    """
    Regret with the CONDITION CHOICE held out as well as the threshold.

    The naive regret elsewhere in this script picks the F1-best and cost-best
    condition using all 15 folds, then measures the gap between them on those
    same 15 folds. Both the winner and the margin are therefore measured on
    data that chose them, which flatters whichever condition happened to suit
    this particular set of folds.

    Here, for every fold in turn:
      * the other 4 folds OF THE SAME REPEAT are the selection set - they
        share no rows with the held-out fold, because within a repeat the 5
        folds partition the data;
      * on that selection set we compute each condition's F1 at 0.5 and its
        cost at its own selection-set-derived threshold;
      * the F1-best and cost-best conditions are chosen from those numbers;
      * both are then scored on the held-out fold, at thresholds that also
        came from the selection set.

    Nothing about the held-out fold influenced which conditions were compared
    or where their thresholds sat. The regret can come out NEGATIVE - meaning
    the F1-chosen condition happened to be cheaper on that fold - which is
    itself informative: it shows how much of the naive regret is selection
    noise rather than a real penalty.
    """
    positions = sorted({(f["repeat"], f["fold"])
                        for f in conditions[names[0]]})
    rows = []

    for repeat, fold in positions:
        selection, held = {}, {}
        for c in names:
            selection[c] = [f for f in conditions[c]
                            if f["repeat"] == repeat and f["fold"] != fold]
            held[c] = next(f for f in conditions[c]
                           if f["repeat"] == repeat and f["fold"] == fold)

        # --- choose, using only the selection folds ---
        thr_sel = {c: pooled_threshold(selection[c], fee) for c in names}
        f1_sel = {c: pooled_f1(selection[c]) for c in names}
        cost_sel = {c: pooled_cost(selection[c], thr_sel[c], fee)
                    for c in names}

        f1_pick = max(names, key=lambda c: f1_sel[c])
        cost_pick = min(names, key=lambda c: cost_sel[c])

        # --- measure, on the untouched held-out fold ---
        cost_f1_pick = pooled_cost([held[f1_pick]], thr_sel[f1_pick], fee)
        cost_cost_pick = pooled_cost([held[cost_pick]], thr_sel[cost_pick], fee)

        rows.append({
            "repeat": repeat, "fold": fold,
            "f1_pick": f1_pick, "cost_pick": cost_pick,
            "same_pick": f1_pick == cost_pick,
            "cost_f1_pick": cost_f1_pick,
            "cost_cost_pick": cost_cost_pick,
            "regret": cost_f1_pick - cost_cost_pick,
        })
    return rows


def rank_desc(values):
    """Rank so that 1 = best, for a higher-is-better quantity."""
    order = np.argsort(-np.asarray(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def rank_asc(values):
    """Rank so that 1 = best, for a lower-is-better quantity."""
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def main() -> None:
    print("=" * 100)
    print("05_cost.py - cost evaluation across every model x strategy condition")
    print("=" * 100)

    conditions = discover_conditions()
    names = sorted(conditions)
    n_rows = len(conditions[names[0]][0]["y_true"])
    n_folds = len(conditions[names[0]])

    print(f"\n  conditions : {len(names)}  ({', '.join(names)})")
    print(f"  folds each : {n_folds} of {n_rows:,} rows")
    print(f"  fees       : {', '.join(f'${f:.2f}' for f in FEES)}")
    print(f"  all costs reported per {COST_BASIS:,} transactions")
    print(f"\n  Thresholds are chosen on the other folds of the same repeat")
    print(f"  and scored on the held-out fold - never on the rows they are")
    print(f"  graded against.")

    per_fee = {}
    for fee in FEES:
        scored = {c: score_condition(conditions[c], fee) for c in names}

        f1_vals = [scored[c]["f1_05"] for c in names]
        cost_vals = [scored[c]["cost_opt"] for c in names]
        auprc_vals = [scored[c]["auprc"] for c in names]

        f1_rank = rank_desc(f1_vals)       # 1 = highest F1
        cost_rank = rank_asc(cost_vals)    # 1 = cheapest
        auprc_rank = rank_desc(auprc_vals)

        rho_f1, p_f1 = spearmanr(f1_rank, cost_rank)
        rho_auprc, p_auprc = spearmanr(auprc_rank, cost_rank)

        f1_best = names[int(np.argmax(f1_vals))]
        cost_best = names[int(np.argmin(cost_vals))]
        auprc_best = names[int(np.argmax(auprc_vals))]
        same_winner = f1_best == cost_best

        # Regret: pick the condition with the best F1, operate it at its own
        # held-out cost-minimising threshold, and compare with the condition
        # that was actually cheapest. This is the GENEROUS reading - the
        # F1-chosen model still gets a well-tuned threshold.
        regret = scored[f1_best]["cost_opt"] - scored[cost_best]["cost_opt"]
        regret_pct = regret / scored[cost_best]["cost_opt"]

        print("\n" + "=" * 100)
        print(f"REVIEW FEE ${fee:.2f}   (cost per {COST_BASIS:,} transactions)")
        print("=" * 100)
        print(f"{'condition':>22} {'AUPRC':>8} {'F1@0.5':>8} {'cost@0.5':>12} | "
              f"{'thr':>8} {'F1@thr':>8} {'cost@thr':>12} "
              f"{'F1 rk':>6} {'$ rk':>5}")
        print("-" * 100)
        for i, c in enumerate(names):
            v = scored[c]
            print(f"{c:>22} {v['auprc']:>8.4f} {v['f1_05']:>8.4f} "
                  f"${v['cost_05']:>11,.0f} | {v['thr']:>8.4f} "
                  f"{v['f1_opt']:>8.4f} ${v['cost_opt']:>11,.0f} "
                  f"{f1_rank[i]:>6} {cost_rank[i]:>5}")
        print("-" * 100)
        print(f"  F1-best    : {f1_best}  (F1 {max(f1_vals):.4f}, "
              f"cost ${scored[f1_best]['cost_opt']:,.0f})")
        print(f"  AUPRC-best : {auprc_best}  (AUPRC {max(auprc_vals):.4f})")
        print(f"  cost-best  : {cost_best}  "
              f"(cost ${min(cost_vals):,.0f})")
        print(f"  same winner under F1 and cost? "
              f"{'YES' if same_winner else 'NO'}")
        print(f"  Spearman rho, F1 rank vs cost rank    : {rho_f1:+.4f}  "
              f"(p = {p_f1:.4f})")
        print(f"  Spearman rho, AUPRC rank vs cost rank : {rho_auprc:+.4f}  "
              f"(p = {p_auprc:.4f})")
        print(f"  REGRET (naive) of choosing on F1 : ${regret:,.0f} per "
              f"{COST_BASIS:,} ({regret_pct:+.1%} vs the cost-best condition)")

        # --- the same quantity, with the condition choice held out too ---
        nested = nested_regret(conditions, names, fee)
        n_regrets = np.array([r["regret"] for r in nested])
        n_mean = float(n_regrets.mean())
        n_sd = float(n_regrets.std(ddof=1))
        n_same = sum(r["same_pick"] for r in nested)
        n_negative = int((n_regrets < 0).sum())
        f1_picks = {}
        cost_picks = {}
        for r in nested:
            f1_picks[r["f1_pick"]] = f1_picks.get(r["f1_pick"], 0) + 1
            cost_picks[r["cost_pick"]] = cost_picks.get(r["cost_pick"], 0) + 1

        print(f"  REGRET (nested, condition chosen on held-out folds):")
        print(f"      ${n_mean:,.0f} per {COST_BASIS:,}  "
              f"(sd ${n_sd:,.0f} across {len(nested)} held-out folds)")
        print(f"      selections agreed on {n_same}/{len(nested)} folds; "
              f"regret was negative on {n_negative}/{len(nested)}")
        print(f"      F1 picked   : "
              f"{', '.join(f'{k} x{v}' for k, v in sorted(f1_picks.items(), key=lambda kv: -kv[1]))}")
        print(f"      cost picked : "
              f"{', '.join(f'{k} x{v}' for k, v in sorted(cost_picks.items(), key=lambda kv: -kv[1]))}")

        per_fee[fee] = {
            "fee": fee, "scored": scored,
            "f1_rank": f1_rank.tolist(), "cost_rank": cost_rank.tolist(),
            "auprc_rank": auprc_rank.tolist(),
            "rho_f1_cost": float(rho_f1), "p_f1_cost": float(p_f1),
            "rho_auprc_cost": float(rho_auprc), "p_auprc_cost": float(p_auprc),
            "f1_best": f1_best, "auprc_best": auprc_best,
            "cost_best": cost_best, "same_winner": bool(same_winner),
            "regret": float(regret), "regret_pct": float(regret_pct),
            "nested_regret_mean": n_mean,
            "nested_regret_sd": n_sd,
            "nested_same_pick": int(n_same),
            "nested_negative": n_negative,
            "nested_n_folds": len(nested),
            "nested_f1_picks": f1_picks,
            "nested_cost_picks": cost_picks,
            "nested_folds": nested,
        }

    # ---- Across the fee range -------------------------------------------
    print("\n" + "=" * 100)
    print("AGREEMENT BETWEEN METRIC RANKINGS AND COST RANKING, ACROSS FEES")
    print("=" * 100)
    print(f"{'fee':>8} {'rho(F1)':>9} {'rho(AUPRC)':>11} {'F1-best':>20} "
          f"{'cost-best':>20} {'same?':>6} {'regret':>11} {'nested':>11} "
          f"{'(sd)':>10}")
    print("-" * 100)
    for fee in FEES:
        r = per_fee[fee]
        print(f"${fee:>7.2f} {r['rho_f1_cost']:>+9.4f} "
              f"{r['rho_auprc_cost']:>+11.4f} {r['f1_best']:>20} "
              f"{r['cost_best']:>20} "
              f"{'YES' if r['same_winner'] else 'NO':>6} "
              f"${r['regret']:>10,.0f} ${r['nested_regret_mean']:>10,.0f} "
              f"${r['nested_regret_sd']:>9,.0f}")
    print("-" * 100)

    rho_f1_all = np.array([per_fee[f]["rho_f1_cost"] for f in FEES])
    rho_ap_all = np.array([per_fee[f]["rho_auprc_cost"] for f in FEES])
    agree = sum(per_fee[f]["same_winner"] for f in FEES)
    regrets = np.array([per_fee[f]["regret"] for f in FEES])

    print(f"\n  rho = +1 means the two rankings are identical, 0 means")
    print(f"  unrelated, -1 means exactly reversed.")
    print(f"\n  F1 vs cost    : mean rho {rho_f1_all.mean():+.4f}, "
          f"range {rho_f1_all.min():+.4f} to {rho_f1_all.max():+.4f}")
    print(f"  AUPRC vs cost : mean rho {rho_ap_all.mean():+.4f}, "
          f"range {rho_ap_all.min():+.4f} to {rho_ap_all.max():+.4f}")
    print(f"  same winner under both criteria at "
          f"{agree} of {len(FEES)} fees")
    nested_means = np.array([per_fee[f]["nested_regret_mean"] for f in FEES])
    print(f"\n  REGRET, TWO WAYS")
    print(f"    naive  : ${regrets.min():,.0f} to ${regrets.max():,.0f} "
          f"per {COST_BASIS:,}")
    print(f"             conditions chosen AND measured on the same 15 folds")
    print(f"    nested : ${nested_means.min():,.0f} to "
          f"${nested_means.max():,.0f} per {COST_BASIS:,}")
    print(f"             conditions chosen on 4 folds, measured on the 5th")
    shrink = 1 - nested_means.mean() / regrets.mean()
    print(f"\n    The nested figure is {abs(shrink):.0%} "
          f"{'SMALLER' if shrink > 0 else 'LARGER'} on average. The gap is the")
    print(f"    part of naive regret that came from choosing the winners on")
    print(f"    the same folds that then measured the gap between them.")
    total_same = sum(per_fee[f]["nested_same_pick"] for f in FEES)
    total_folds = sum(per_fee[f]["nested_n_folds"] for f in FEES)
    total_neg = sum(per_fee[f]["nested_negative"] for f in FEES)
    print(f"\n    Across all {total_folds} held-out evaluations, the F1 and")
    print(f"    cost criteria picked the same condition {total_same} times")
    print(f"    ({total_same / total_folds:.0%}), and nested regret came out")
    print(f"    negative {total_neg} times ({total_neg / total_folds:.0%}) -")
    print(f"    folds where choosing on F1 happened to be cheaper.")

    # ---- Threshold-selection leakage, once -------------------------------
    print("\n" + "=" * 100)
    print(f"THRESHOLD-SELECTION BIAS AT ${REFERENCE_FEE:.2f}")
    print("=" * 100)
    ref = per_fee[REFERENCE_FEE]["scored"]
    print(f"{'condition':>22} {'held-out':>12} {'in-sample':>12} {'bias':>9}")
    print("-" * 100)
    for c in names:
        honest, leaky = ref[c]["cost_opt"], ref[c]["cost_opt_leaky"]
        print(f"{c:>22} ${honest:>11,.0f} ${leaky:>11,.0f} "
              f"{(honest - leaky) / honest:>+8.1%}")
    print("-" * 100)
    print(f"  In-sample thresholds understate cost because they were chosen")
    print(f"  using the labels they are then scored against. Every number")
    print(f"  elsewhere in this script uses the held-out column.")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / "cost_comparison_all_conditions.json"
    out.write_text(json.dumps({
        "cost_basis": COST_BASIS,
        "fees": FEES,
        "conditions": names,
        "threshold_selection": "held out: other folds of the same repeat",
        "results": {str(f): {k: v for k, v in per_fee[f].items()}
                    for f in FEES},
    }, indent=2, default=float))
    print(f"\n  wrote {out.relative_to(PROJECT_ROOT)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
