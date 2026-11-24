"""
08_rq3_spread.py - FIN 611 fraud detection research project

RQ3: does the gap between metric-based and cost-based model selection depend
     on how UNEQUAL the fraud amounts are?

  The intuition. If every fraud stole exactly the same amount, then missing a
  fraud would cost the same whichever fraud you missed, and minimising cost
  would collapse into maximising recall - something F1 and AUPRC already
  reward. The reason cost-based selection can disagree with them is that
  frauds differ in size, so WHICH frauds a model catches matters as much as
  how many. Push the amounts together and the disagreement should shrink;
  pull them apart and it should grow.

  This script tests that by varying the inequality of the fraud amounts while
  holding everything else fixed.

NOTHING IS RETRAINED. NOTHING IS REFITTED.

  The 180 saved .npz files in results/predictions/ are used exactly as they
  are. The models, the features, the folds, the hyperparameters and the
  predicted probabilities are all untouched - the same numbers 05_cost.py
  read. The ONLY thing that changes is the dollar value attached to each
  fraud when the cost of missing it is computed.

  That is what makes this a clean experiment. Any change in the results comes
  from the amount distribution and from nothing else, because nothing else
  moved. A model that was better at ranking is still exactly as good at
  ranking; it is only the price list that changed.

HOW THE FOUR AMOUNT VECTORS ARE BUILT

  A power transform, then a rescale:

      amount' = c * amount ** gamma,      c chosen so mean(amount') = mean(amount)

  gamma controls inequality. Below 1 it pulls values toward each other -
  raising a number to the power 0.3 moves $1 and $1,000 much closer together
  in relative terms. Above 1 it drives them apart. gamma = 1 is the identity,
  so the "real" version reproduces the original amounts bit-for-bit.

  THE RESCALE IS THE POINT OF THE DESIGN. Without it, raising amounts to the
  power 4 would multiply the total money at risk by an enormous factor, and
  every cost would rise for that reason alone - the comparison would measure
  the scale change, not the inequality change. Pinning the mean pins the
  TOTAL fraud exposure: all four versions have the same 2,084 frauds worth
  the same total dollars, distributed differently. Only the spread varies.

  Gini is scale-invariant, so the rescale does not touch it: inequality is
  set by gamma alone, and the constant c only restores the level.

  Legitimate transaction amounts are left alone. They never enter the cost
  model - a flagged legitimate transaction costs the review fee regardless of
  its size, and an unflagged one costs nothing - so transforming them would
  change no number in this script.

WHY THE TRANSFORM IS DEFINED ON THE POOLED POPULATION, NOT PER FOLD

  The outer design is RepeatedStratifiedKFold(5 x 3), so each transaction
  appears in three different held-out folds. If the rescaling constant were
  computed inside each fold, the same transaction would be worth different
  amounts in different folds, and the cost comparison would be incoherent.

  So the transform is fitted ONCE on the pooled population of 2,084 fraud
  amounts - obtained by concatenating the five folds of a single repeat,
  which partition the full 400,000 rows exactly - and the resulting
  deterministic map is then applied everywhere. A given fraud is worth the
  same amount in every fold it appears in.

WHAT IS MEASURED

  Mean NESTED regret and its standard deviation, at each of the five fees,
  for each amount version. Nested regret is the quantity 05_cost.py defines
  and this script imports rather than re-implements: for each of the 15 outer
  folds, the condition AND its threshold are chosen on the other four folds
  of the same repeat, then both the F1-chosen and cost-chosen conditions are
  scored on the held-out fold. It is the honest version of "what does
  choosing on F1 cost you", with no selection measured on the data that made
  it.

  Because the real version reproduces the original amounts exactly, its row
  must reproduce the nested regret already stored in
  cost_comparison_all_conditions.json. The script asserts that as a
  known-answer check on the whole harness.

Run from anywhere:  python src/08_rq3_spread.py     (takes roughly 5 minutes)
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
OUT_PATH = TABLE_DIR / "rq3_spread.json"
REFERENCE_PATH = TABLE_DIR / "cost_comparison_all_conditions.json"

# The four amount versions, in ascending order of inequality. The gamma
# values were chosen by sweeping gamma against the Gini coefficient of the
# real amounts and picking round numbers that spread Gini roughly evenly over
# the achievable range:
#
#     gamma 0.30 -> Gini 0.204    squeezed
#     gamma 1.00 -> Gini 0.419    real (the actual data)
#     gamma 2.00 -> Gini 0.550    stretched
#     gamma 4.00 -> Gini 0.670    very stretched
#
# That is a 3.3x span in Gini, wide enough that any dependence of regret on
# inequality should be visible, while keeping the real data in the interior
# of the range rather than at an endpoint.
GAMMAS = {
    "squeezed": 0.30,
    "real": 1.00,
    "stretched": 2.00,
    "very_stretched": 4.00,
}


def load_cost_module():
    """
    Import src/05_cost.py as a module.

    The cost model, the threshold search and above all nested_regret are
    taken from there rather than copied, so "exactly as 05_cost.py does it"
    is guaranteed by construction instead of by careful re-implementation.
    The leading digits in the filename make it an invalid identifier, so it
    has to be loaded by path.
    """
    path = PROJECT_ROOT / "src" / "05_cost.py"
    spec = importlib.util.spec_from_file_location("cost_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gini(values) -> float:
    """
    Gini coefficient of a non-negative vector.

    The definition is the mean absolute difference between all pairs,
    normalised:

        G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mean)

    which is 0 when every value is identical and approaches 1 as one value
    takes everything. Computing it that way needs an n x n matrix - 2,084^2
    is survivable but wasteful - so the equivalent sorted form is used:

        G = 2 * sum(i * x_sorted_i) / (n * sum(x)) - (n + 1) / n,   i = 1..n

    gini_reference() below is the literal double sum, and the script checks
    the two agree before using the fast one.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(2.0 * np.sum(index * x) / (n * total) - (n + 1.0) / n)


def gini_reference(values) -> float:
    """Gini straight from the definition, for validating gini()."""
    x = np.asarray(values, dtype=float)
    n = x.size
    return float(np.abs(x[:, None] - x[None, :]).sum()
                 / (2.0 * n * n * x.mean()))


def fraud_population(conditions, names) -> np.ndarray:
    """
    Every fraud amount in the dataset, once each.

    Within one repeat the five folds partition the full 400,000 rows, so
    concatenating them recovers the whole dataset and its 2,084 frauds with
    no duplication. Repeat 1 of the first condition is as good as any other.

    The function also verifies that every condition and every repeat sees the
    same fraud amounts. They must: 04_grid.py reuses one fold assignment for
    all conditions. If they ever diverged, the transform fitted here would
    not be the transform the other conditions need, so it is worth asserting
    rather than assuming.
    """
    def pop_for(condition, repeat):
        folds = sorted([f for f in conditions[condition]
                        if f["repeat"] == repeat], key=lambda f: f["fold"])
        y = np.concatenate([f["y_true"] for f in folds])
        a = np.concatenate([f["amt"] for f in folds])
        return np.sort(a[y == 1])

    reference = pop_for(names[0], 1)
    for condition in names:
        for repeat in (1, 2, 3):
            other = pop_for(condition, repeat)
            if not np.array_equal(reference, other):
                raise SystemExit(
                    f"fraud amounts differ between conditions/repeats "
                    f"({condition}, repeat {repeat}) - the folds are not the "
                    f"shared assignment this script assumes")
    return reference


def transform_factory(population: np.ndarray, gamma: float, target_mean: float):
    """
    Build the deterministic map amount -> c * amount**gamma.

    c is solved once from the pooled population so that the transformed
    population has exactly the target mean. The returned function is then
    applied to individual folds, which is what keeps a given transaction
    worth the same amount wherever it appears.
    """
    powered = population ** gamma
    c = target_mean / powered.mean()

    def apply(amounts):
        return c * (np.asarray(amounts, dtype=float) ** gamma)

    return apply, float(c)


def retag_amounts(conditions, apply_fn) -> None:
    """
    Replace each fold's fraud amounts in place, from the pristine originals.

    Every fold keeps its untouched amounts in "amt_real", so each version is
    derived from the original data rather than from the previous version -
    four successive transforms would otherwise compound.

    Only rows with y_true == 1 are rewritten. Legitimate amounts never reach
    the cost model, so leaving them alone keeps the arrays honest about what
    this script actually varies.
    """
    for folds in conditions.values():
        for fold in folds:
            amounts = fold["amt_real"].astype(float).copy()
            is_fraud = fold["y_true"] == 1
            amounts[is_fraud] = apply_fn(amounts[is_fraud])
            fold["amt"] = amounts


def aggregate_winners(cost, conditions, names, fee, f1_means):
    """
    The F1-best and cost-best conditions over all 15 folds, as 05_cost.py
    defines them: the mean held-out cost per condition, and the mean F1 at
    the default 0.5 threshold.

    F1 at a fixed threshold does not involve amounts at all, so f1_means is
    computed once by the caller and reused for every version. That is not
    merely an optimisation - it is the mechanism this whole script probes.
    The F1-chosen condition CANNOT respond to the amount distribution, while
    the cost-chosen one can.
    """
    cost_means = []
    for condition in names:
        folds = conditions[condition]
        per_fold = []
        for fold in folds:
            others = [f for f in folds
                      if f["repeat"] == fold["repeat"]
                      and f["name"] != fold["name"]]
            threshold = cost.pooled_threshold(others, fee)
            per_fold.append(cost.evaluate_at(fold, threshold, fee)["cost"])
        cost_means.append(float(np.mean(per_fold)))

    return {
        "f1_best": names[int(np.argmax(f1_means))],
        "cost_best": names[int(np.argmin(cost_means))],
        "cost_best_value": float(np.min(cost_means)),
        "f1_best_cost": float(cost_means[int(np.argmax(f1_means))]),
        "cost_per_condition": dict(zip(names, cost_means)),
    }


def main() -> None:
    cost = load_cost_module()
    fees = list(cost.FEES)

    print("=" * 100)
    print("08_rq3_spread.py - RQ3: does fraud-amount inequality drive the")
    print("                   metric-vs-cost selection gap?")
    print("=" * 100)
    print("  Nothing is retrained. The saved predictions, features, folds and")
    print("  probabilities are used as-is; only the dollar value attached to")
    print("  each fraud changes.")

    conditions = cost.discover_conditions()
    names = sorted(conditions)

    # Keep a pristine copy of every fold's amounts to transform from.
    for folds in conditions.values():
        for fold in folds:
            fold["amt_real"] = fold["amt"].astype(float).copy()

    population = fraud_population(conditions, names)
    target_mean = float(population.mean())

    print(f"\n  conditions        : {len(names)}")
    print(f"  folds each        : {len(conditions[names[0]])}")
    print(f"  fees              : "
          f"{', '.join(f'${f:.2f}' for f in fees)}")
    print(f"  frauds in dataset : {population.size:,}")
    print(f"  mean fraud amount : ${target_mean:.6f}  "
          f"(the ${target_mean:,.2f} every version is rescaled to)")
    print(f"  total at risk     : ${population.sum():,.2f}  "
          f"(identical in all four versions, by construction)")

    # Validate the fast Gini against the definition before trusting it.
    check_fast = gini(population)
    check_ref = gini_reference(population)
    print(f"\n  Gini self-check: sorted formula {check_fast:.10f} vs "
          f"double sum {check_ref:.10f}")
    if abs(check_fast - check_ref) > 1e-10:
        raise SystemExit("gini() disagrees with its own definition")
    print(f"  agree to 1e-10, so the fast formula is used below.")

    # F1 at 0.5 is amount-independent: computed once, reused for all versions.
    f1_means = [
        float(np.mean([
            cost.f1_score(f["y_true"],
                          (f["y_prob"] >= cost.DEFAULT_THRESHOLD).astype(int),
                          zero_division=0)
            for f in conditions[c]]))
        for c in names
    ]

    rows = []
    versions = {}

    for version, gamma in GAMMAS.items():
        apply_fn, c = transform_factory(population, gamma, target_mean)
        retag_amounts(conditions, apply_fn)

        transformed = apply_fn(population)
        g = gini(transformed)

        # The rescale must land the mean exactly on the original.
        drift = abs(float(transformed.mean()) - target_mean)
        if drift > 1e-6:
            raise SystemExit(
                f"{version}: rescaled mean is off by {drift:.3e}")

        # gamma = 1 is the identity, so "real" must be bit-identical to the
        # data on disk. If it is not, the transform machinery is perturbing
        # something it should not.
        if gamma == 1.0:
            for folds in conditions.values():
                for fold in folds:
                    if not np.array_equal(fold["amt"], fold["amt_real"]):
                        raise SystemExit(
                            "real version is not bit-identical to the saved "
                            "amounts - the identity transform is not the "
                            "identity")

        print("\n" + "=" * 100)
        print(f"VERSION {version}   gamma = {gamma}   Gini = {g:.4f}")
        print("=" * 100)
        print(f"  amount' = {c:.6g} * amount ** {gamma}")
        print(f"  mean ${transformed.mean():.6f}   "
              f"min ${transformed.min():,.4f}   "
              f"median ${np.median(transformed):,.2f}   "
              f"max ${transformed.max():,.2f}   "
              f"max/mean {transformed.max() / transformed.mean():.1f}x")
        if gamma == 1.0:
            print(f"  (identity transform - bit-identical to the saved data)")

        versions[version] = {
            "version": version, "gamma": gamma, "gini": g, "scale_c": c,
            "mean": float(transformed.mean()),
            "min": float(transformed.min()),
            "median": float(np.median(transformed)),
            "max": float(transformed.max()),
            "total_at_risk": float(transformed.sum()),
        }

        print(f"\n{'fee':>8} {'mean regret':>13} {'sd':>12} "
              f"{'cost-best':>20} {'F1-best':>20} {'neg':>6} {'same':>6}")
        print("-" * 100)
        for fee in fees:
            nested = cost.nested_regret(conditions, names, fee)
            regrets = np.array([r["regret"] for r in nested], dtype=float)
            agg = aggregate_winners(cost, conditions, names, fee, f1_means)

            row = {
                "version": version,
                "gamma": gamma,
                "gini": g,
                "fee": fee,
                "mean_regret": float(regrets.mean()),
                "sd_regret": float(regrets.std(ddof=1)),
                "cost_best": agg["cost_best"],
                "f1_best": agg["f1_best"],
                "cost_best_value": agg["cost_best_value"],
                "f1_best_cost": agg["f1_best_cost"],
                "same_winner": bool(agg["cost_best"] == agg["f1_best"]),
                "n_folds": int(regrets.size),
                "n_negative": int((regrets < 0).sum()),
                "n_same_pick": int(sum(r["same_pick"] for r in nested)),
                "fold_regrets": regrets.tolist(),
                "cost_per_condition": agg["cost_per_condition"],
            }
            rows.append(row)

            print(f"${fee:>7.2f} ${row['mean_regret']:>12,.0f} "
                  f"${row['sd_regret']:>11,.0f} {row['cost_best']:>20} "
                  f"{row['f1_best']:>20} {row['n_negative']:>4}/15 "
                  f"{row['n_same_pick']:>4}/15")
        print("-" * 100)

    # ---- known-answer check on the real version -------------------------
    print("\n" + "=" * 100)
    print("KNOWN-ANSWER CHECK: the real version must reproduce 05_cost.py")
    print("=" * 100)
    if REFERENCE_PATH.exists():
        reference = json.loads(REFERENCE_PATH.read_text())
        print(f"{'fee':>8} {'this script':>14} {'05_cost.py':>14} "
              f"{'sd here':>12} {'sd there':>12} {'match':>7}")
        print("-" * 100)
        all_match = True
        for fee in fees:
            mine = next(r for r in rows
                        if r["version"] == "real" and r["fee"] == fee)
            theirs = reference["results"][str(fee)]
            ok = (mine["mean_regret"] == theirs["nested_regret_mean"]
                  and mine["sd_regret"] == theirs["nested_regret_sd"])
            all_match &= ok
            print(f"${fee:>7.2f} ${mine['mean_regret']:>13,.4f} "
                  f"${theirs['nested_regret_mean']:>13,.4f} "
                  f"${mine['sd_regret']:>11,.4f} "
                  f"${theirs['nested_regret_sd']:>11,.4f} "
                  f"{'EXACT' if ok else 'DIFFERS':>7}")
        print("-" * 100)
        if all_match:
            print("  All five match exactly - the harness changes nothing "
                  "but the amounts.")
        else:
            print("  MISMATCH - investigate before using these results.")
    else:
        print("  (cost_comparison_all_conditions.json absent; check skipped)")

    # ---- the headline table ---------------------------------------------
    order = sorted(GAMMAS, key=lambda v: versions[v]["gini"])

    print("\n" + "=" * 100)
    print(f"GINI vs MEAN NESTED REGRET  (dollars per "
          f"{cost.COST_BASIS:,} transactions)")
    print("=" * 100)
    print(f"{'version':>16} {'gamma':>6} {'gini':>7} |"
          + "".join(f"{f'${f:.2f}':>11}" for f in fees))
    print("-" * 100)
    for version in order:
        line = (f"{version:>16} {GAMMAS[version]:>6.2f} "
                f"{versions[version]['gini']:>7.4f} |")
        for fee in fees:
            r = next(x for x in rows
                     if x["version"] == version and x["fee"] == fee)
            cell = f"${r['mean_regret']:,.0f}"
            line += f"{cell:>11}"
        print(line)
    print("-" * 100)

    print(f"\n  standard deviations across the 15 held-out folds")
    print(f"{'version':>16} {'gamma':>6} {'gini':>7} |"
          + "".join(f"{f'${f:.2f}':>11}" for f in fees))
    print("-" * 100)
    for version in order:
        line = (f"{version:>16} {GAMMAS[version]:>6.2f} "
                f"{versions[version]['gini']:>7.4f} |")
        for fee in fees:
            r = next(x for x in rows
                     if x["version"] == version and x["fee"] == fee)
            cell = f"${r['sd_regret']:,.0f}"
            line += f"{cell:>11}"
        print(line)
    print("-" * 100)

    # ---- is regret monotone in Gini? ------------------------------------
    print("\n" + "=" * 100)
    print("DOES REGRET RISE WITH INEQUALITY?")
    print("=" * 100)
    ginis = np.array([versions[v]["gini"] for v in order])
    print(f"  Only {len(order)} amount versions, so this is a direction "
          f"check, not a test.")
    print(f"\n{'fee':>8} {'monotone in Gini?':>20} "
          f"{'squeezed':>12} {'very_stretched':>16} {'ratio':>9}")
    print("-" * 100)
    for fee in fees:
        series = np.array([
            next(x["mean_regret"] for x in rows
                 if x["version"] == v and x["fee"] == fee)
            for v in order])
        rising = bool(np.all(np.diff(series) > 0))
        falling = bool(np.all(np.diff(series) < 0))
        shape = ("rising" if rising else "falling" if falling
                 else "non-monotone")
        lo, hi = series[0], series[-1]
        ratio = hi / lo if lo != 0 else float("inf")
        print(f"${fee:>7.2f} {shape:>20} ${lo:>11,.0f} ${hi:>15,.0f} "
              f"{ratio:>8.2f}x")
    print("-" * 100)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "research_question": "RQ3: does the metric-vs-cost selection gap "
                             "depend on the inequality of fraud amounts?",
        "design": "No retraining. Saved per-fold predictions are reused "
                  "unchanged; only the dollar value attached to each fraud "
                  "varies, via amount' = c * amount**gamma with c chosen so "
                  "the mean is preserved exactly.",
        "retrained": False,
        "transform": "c * amount ** gamma, fitted once on the pooled "
                     "population of fraud amounts so a transaction is worth "
                     "the same in every fold it appears in",
        "mean_preserved": target_mean,
        "note": "Gini is scale-invariant, so inequality is set by gamma "
                "alone and the constant c only restores the level. "
                "Legitimate amounts are untouched: they never enter the "
                "cost model. F1 at a fixed threshold is amount-independent, "
                "so the F1-selected condition is identical in all four "
                "versions by construction.",
        "cost_basis": cost.COST_BASIS,
        "units": f"dollars per {cost.COST_BASIS:,} transactions",
        "threshold_selection": "held out: other folds of the same repeat",
        "condition_selection": "nested: chosen on the other folds of the "
                               "same repeat, scored on the held-out fold "
                               "(05_cost.py nested_regret, imported)",
        "n_frauds": int(population.size),
        "total_at_risk": float(population.sum()),
        "conditions": names,
        "fees": fees,
        "gammas": GAMMAS,
        "versions": versions,
        "rows": rows,
    }, indent=2, default=float))
    print(f"\n  wrote {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
