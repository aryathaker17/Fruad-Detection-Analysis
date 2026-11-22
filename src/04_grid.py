"""
04_grid.py - FIN 611 fraud detection research project

WHAT THIS DOES
  Runs every (model x imbalance strategy) pair as a SEPARATE, FIXED CONDITION,
  and inside each one tunes only the MODEL's hyperparameters by nested
  cross-validation scored on average_precision (AUPRC).

  The imbalance strategy is NOT in the search space. That is deliberate and it
  is the whole design of this script. If you let a search choose between SMOTE,
  undersampling and nothing, it reports one winner and you learn nothing about
  the others - and the winner varies fold to fold when the candidates are
  close, which tells you less than it appears to. Fixing the strategy per
  condition means every strategy is measured under identical, honest
  conditions and all of them get reported.

THE DESIGN
    conditions = MODELS x STRATEGIES          (reported separately)
      for each condition:
        OUTER  RepeatedStratifiedKFold(5 x 3) = 15 folds, for scoring
          INNER  StratifiedKFold(3) + RandomizedSearchCV(average_precision)
                 over MODEL hyperparameters only

  The inner loop never sees the outer test fold, so the outer score measures
  the whole fit-and-tune PROCEDURE on data it has never touched. Scaling,
  encoding and resampling all live in an imblearn Pipeline, so each inner fit
  refits them from scratch on its own training rows.

  imblearn's Pipeline, not sklearn's: it applies samplers during fit() and
  SKIPS them at predict() time. You resample what the model learns from, never
  what you score on.

OUTPUT
  results/predictions/<model>__<strategy>_r<repeat>_f<fold>.npz
  holding y_true, y_prob, amt, test_index, repeat, fold, best_params (JSON)
  and inner_best_score. The condition is in the filename; the hyperparameters
  the inner loop chose for that fold are in the file.

Run from anywhere:  python src/04_grid.py
Takes roughly 20 minutes on 10 cores.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import loguniform, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV,
                                     RepeatedStratifiedKFold, StratifiedKFold)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Feature engineering is shared with the baseline; import rather than duplicate.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "baseline", Path(__file__).resolve().parent / "03_baseline.py")
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)
make_features = _baseline.make_features
NUMERIC_FEATURES = _baseline.NUMERIC_FEATURES
CATEGORICAL_FEATURES = _baseline.CATEGORICAL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "sample.parquet"
PRED_DIR = PROJECT_ROOT / "results" / "predictions"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
MANIFEST_PATH = TABLE_DIR / "grid_manifest.json"

N_SPLITS_OUTER = 5
N_REPEATS_OUTER = 3
N_SPLITS_INNER = 3
N_ITER_SEARCH = 15
SEARCH_SCORING = "average_precision"
RANDOM_SEED = 611

# ---------------------------------------------------------------------------
# Models, and the hyperparameters tuned for each
# ---------------------------------------------------------------------------
# Only MODEL parameters appear here. Nothing about class balance does.
MODELS = {
    "logreg": {
        "estimator": lambda: LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED),
        "params": {
            # Log-uniform because the order of magnitude is what matters:
            # 0.01 -> 0.1 is as big a step as 10 -> 100.
            "clf__C": loguniform(1e-3, 1e2),
        },
    },
    "xgb": {
        "estimator": lambda: XGBClassifier(
            tree_method="hist", eval_metric="aucpr", n_jobs=1,
            random_state=RANDOM_SEED),
        "params": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [3, 5, 7, 9],
            "clf__learning_rate": loguniform(0.01, 0.3),
            "clf__subsample": uniform(0.6, 0.4),        # 0.6 to 1.0
            "clf__colsample_bytree": uniform(0.6, 0.4),
            "clf__min_child_weight": [1, 5, 20],
        },
    },
    "rf": {
        # n_jobs=-1 parallelises the forest itself. Note that
        # RandomizedSearchCV below also runs n_jobs=-1, so the two nest.
        "estimator": lambda: RandomForestClassifier(
            n_jobs=-1, random_state=RANDOM_SEED),
        "params": {
            # 54 combinations in all, so n_iter=15 samples 15 distinct ones.
            "clf__n_estimators": [100, 200, 400],
            "clf__max_depth": [None, 10, 20],      # None = grow unrestricted
            "clf__min_samples_leaf": [1, 5, 20],
            "clf__max_features": ["sqrt", 0.3],
        },
    },
}

# ---------------------------------------------------------------------------
# Imbalance strategies - FIXED per condition, never searched
# ---------------------------------------------------------------------------
# sampling_strategy is the minority:majority ratio AFTER sampling. The data
# starts near 0.005, so 0.10 is a substantial rebalance either way.
STRATEGIES = {
    "none": {
        "sampler": lambda: "passthrough",
        "reweight": False,
        "note": "no resampling, no reweighting",
    },
    "class_weight": {
        "sampler": lambda: "passthrough",
        "reweight": True,
        "note": "cost-sensitive learning: errors on the minority class are "
                "weighted up in the loss, data untouched",
    },
    "smote": {
        "sampler": lambda: SMOTE(sampling_strategy=0.10, k_neighbors=5,
                                 random_state=RANDOM_SEED),
        "reweight": False,
        "note": "synthesise minority rows by interpolating between neighbours",
    },
    "undersample": {
        "sampler": lambda: RandomUnderSampler(sampling_strategy=0.10,
                                              random_state=RANDOM_SEED),
        "reweight": False,
        "note": "discard majority rows at random until the ratio is reached",
    },
}


def describe_space(dist) -> str:
    """
    Human-readable description of one search-space entry.

    RandomizedSearchCV accepts either a list (sampled uniformly) or a scipy
    frozen distribution (sampled from). Printing repr() of a frozen
    distribution gives an opaque object address, so unpack the useful part.
    """
    if isinstance(dist, list):
        return f"choice of {len(dist)}   {dist}"
    name = getattr(getattr(dist, "dist", None), "name", "")
    args = getattr(dist, "args", ())
    if name == "loguniform" and len(args) == 2:
        lo, hi = args
        decades = np.log10(hi / lo)
        unit = "order" if abs(decades - 1) < 0.05 else "orders"
        return (f"log-uniform [{lo:g}, {hi:g}]   "
                f"({decades:.1f} {unit} of magnitude)")
    if name == "uniform" and len(args) == 2:
        loc, scale = args
        return f"uniform [{loc:g}, {loc + scale:g}]"
    return repr(dist)


def print_search_spaces() -> None:
    """
    Print exactly what the inner loop searches over.

    Worth printing on every run: a reproducibility section needs the search
    space, not just the winning values, and a space defined in code drifts out
    of sync with a space written down in a paper.
    """
    print("\n" + "=" * 92)
    print("HYPERPARAMETER SEARCH SPACES")
    print("=" * 92)
    print(f"  RandomizedSearchCV(n_iter={N_ITER_SEARCH}) scoring "
          f"'{SEARCH_SCORING}', inner cv StratifiedKFold({N_SPLITS_INNER}),")
    print(f"  reseeded per outer fold. MODEL parameters only - the imbalance")
    print(f"  strategy is fixed per condition and never searched.\n")

    for model_key, cfg in MODELS.items():
        params = cfg["params"]
        n_discrete = 1
        continuous = False
        for d in params.values():
            if isinstance(d, list):
                n_discrete *= len(d)
            else:
                continuous = True
        size = ("a continuous space" if continuous
                else f"{n_discrete} combinations")
        print(f"  {model_key}  ({len(params)} parameter"
              f"{'s' if len(params) != 1 else ''}; "
              f"{N_ITER_SEARCH} draws from {size})")
        for key, dist in params.items():
            print(f"    {key.replace('clf__', ''):<20} {describe_space(dist)}")
        if continuous and N_ITER_SEARCH < 10 * len(params):
            print(f"    NOTE: {N_ITER_SEARCH} draws over {len(params)} "
                  f"dimensions is thin. A parameter whose selected")
            print(f"    values end up spanning its whole range may be flat, "
                  f"or may just be undersampled -")
            print(f"    this run cannot distinguish those.")
        print()


def build_estimator(model_key: str, strategy_key: str, y_train) -> ImbPipeline:
    """
    Assemble one condition's pipeline with the strategy baked in.

    The sampler sits AFTER the ColumnTransformer because SMOTE interpolates
    between numeric vectors and cannot work on raw category strings, and
    BEFORE the classifier because its job is to change what the classifier
    learns from.

    Reweighting is applied here, not searched: as scale_pos_weight for
    XGBoost, and as class_weight="balanced" for logistic regression and the
    random forest. All are computed from the TRAINING labels only.
    """
    strategy = STRATEGIES[strategy_key]
    clf = MODELS[model_key]["estimator"]()

    if strategy["reweight"]:
        if model_key == "xgb":
            n_neg = int((y_train == 0).sum())
            n_pos = max(int((y_train == 1).sum()), 1)
            clf.set_params(scale_pos_weight=n_neg / n_pos)
        else:
            clf.set_params(class_weight="balanced")

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"),
         CATEGORICAL_FEATURES),
    ])
    return ImbPipeline([
        ("prep", preprocess),
        ("sampler", strategy["sampler"]()),
        ("clf", clf),
    ])


def jsonable(params: dict) -> dict:
    """Best-params dict with numpy scalars converted for JSON."""
    out = {}
    for k, v in params.items():
        key = k.replace("clf__", "")
        out[key] = v.item() if isinstance(v, np.generic) else v
    return out


def run_condition(model_key, strategy_key, X, y, amounts):
    """Nested CV for one (model, strategy) condition. Returns per-fold rows."""
    outer = RepeatedStratifiedKFold(
        n_splits=N_SPLITS_OUTER, n_repeats=N_REPEATS_OUTER,
        random_state=RANDOM_SEED)
    rows = []

    for i, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        repeat, fold = i // N_SPLITS_OUTER + 1, i % N_SPLITS_OUTER + 1
        t0 = time.time()
        y_train = y[train_idx]

        inner = StratifiedKFold(n_splits=N_SPLITS_INNER, shuffle=True,
                                random_state=RANDOM_SEED + i)
        search = RandomizedSearchCV(
            estimator=build_estimator(model_key, strategy_key, y_train),
            param_distributions=MODELS[model_key]["params"],
            n_iter=N_ITER_SEARCH,
            scoring=SEARCH_SCORING,
            cv=inner,
            random_state=RANDOM_SEED + i,
            n_jobs=-1,
            refit=True,
            error_score=0.0,
        )
        search.fit(X.iloc[train_idx], y_train)

        y_prob = search.predict_proba(X.iloc[test_idx])[:, 1]
        y_true = y[test_idx]

        roc = roc_auc_score(y_true, y_prob)
        pr = average_precision_score(y_true, y_prob)
        flagged = y_prob >= 0.5
        f1 = f1_score(y_true, flagged.astype(int), zero_division=0)
        precision = (flagged & (y_true == 1)).sum() / max(flagged.sum(), 1)
        recall = (flagged & (y_true == 1)).sum() / max(y_true.sum(), 1)

        params_out = jsonable(search.best_params_)
        condition = f"{model_key}__{strategy_key}"
        out_path = PRED_DIR / f"{condition}_r{repeat}_f{fold}.npz"
        np.savez_compressed(
            out_path,
            y_true=y_true, y_prob=y_prob, amt=amounts[test_idx],
            test_index=test_idx, repeat=repeat, fold=fold,
            model=model_key, strategy=strategy_key,
            # Mixed-type params do not fit a numeric array; stored as JSON.
            #   json.loads(np.load(path)["best_params"].item())
            best_params=json.dumps(params_out),
            inner_best_score=float(search.best_score_),
        )

        rows.append({
            "condition": condition, "model": model_key,
            "strategy": strategy_key, "repeat": repeat, "fold": fold,
            "file": out_path.name,
            "roc_auc": float(roc), "pr_auc": float(pr), "f1_at_0.5": float(f1),
            "precision_at_0.5": float(precision),
            "recall_at_0.5": float(recall),
            "best_params": params_out,
            "inner_best_score": float(search.best_score_),
            "seconds": time.time() - t0,
        })
    return rows


def summarise(model_key: str, strategy_key: str, rows: list) -> dict:
    """Mean metrics for one condition across its outer folds."""
    return {
        "model": model_key, "strategy": strategy_key,
        "condition": f"{model_key}__{strategy_key}",
        **{f"{k}_mean": float(np.mean([r[k] for r in rows]))
           for k in ("roc_auc", "pr_auc", "f1_at_0.5",
                     "precision_at_0.5", "recall_at_0.5")},
        "pr_auc_std": float(np.std([r["pr_auc"] for r in rows], ddof=1)),
    }


def manifest_metadata() -> dict:
    """Design fields, identical whether the manifest is rewritten or appended."""
    return {
        "design": "model x strategy conditions; nested CV on model params only",
        "imbalance_strategy_in_search_space": False,
        "outer": f"RepeatedStratifiedKFold({N_SPLITS_OUTER}x{N_REPEATS_OUTER})",
        "inner": f"StratifiedKFold({N_SPLITS_INNER}) + "
                 f"RandomizedSearchCV(n_iter={N_ITER_SEARCH})",
        "inner_scoring": SEARCH_SCORING,
        "random_seed": RANDOM_SEED,
        "models": list(MODELS),
        "strategies": {k: v["note"] for k, v in STRATEGIES.items()},
    }


def _order_key(d: dict):
    """Canonical order: model order, then strategy order, then repeat/fold.

    Sorting on this makes the manifest independent of the order conditions
    happened to be run in, so a file built by appending matches one built in
    a single full pass.
    """
    models, strategies = list(MODELS), list(STRATEGIES)
    m = models.index(d["model"]) if d["model"] in models else len(models)
    s = (strategies.index(d["strategy"]) if d["strategy"] in strategies
         else len(strategies))
    return (m, s, d.get("repeat", 0), d.get("fold", 0))


def append_to_manifest(rows: list, summary: dict) -> tuple:
    """
    Merge one condition into grid_manifest.json, leaving the others alone.

    Re-running a condition REPLACES its previous rows rather than appending a
    second copy, so the file stays one run per condition and the operation is
    safe to repeat.
    """
    condition = summary["condition"]
    if MANIFEST_PATH.exists():
        man = json.loads(MANIFEST_PATH.read_text())
    else:
        man = {"summaries": [], "folds": []}

    replaced = len([r for r in man.get("folds", [])
                    if r["condition"] == condition])
    folds = [r for r in man.get("folds", [])
             if r["condition"] != condition] + rows
    summaries = [t for t in man.get("summaries", [])
                 if t["condition"] != condition] + [summary]

    # Refresh metadata: a newly added model has to show up in "models".
    man.update(manifest_metadata())
    man["summaries"] = sorted(summaries, key=_order_key)
    man["folds"] = sorted(folds, key=_order_key)
    MANIFEST_PATH.write_text(json.dumps(man, indent=2))
    return replaced, len(man["summaries"]), len(man["folds"])


def _show_selected(rows: list) -> None:
    """Compact per-condition view of which hyperparameters were chosen."""
    for key in sorted({k for r in rows for k in r["best_params"]}):
        vals = [r["best_params"][key] for r in rows]
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v in vals)
        if numeric and len({str(v) for v in vals}) > 4:
            arr = np.array(vals, dtype=float)
            print(f"    {key:<20} median {np.median(arr):<12.4g} "
                  f"range {arr.min():.4g} to {arr.max():.4g}")
        else:
            counts = {}
            for v in vals:
                counts[str(v)] = counts.get(str(v), 0) + 1
            shown = "  ".join(f"{v} x{c}" for v, c in
                              sorted(counts.items(), key=lambda kv: -kv[1]))
            print(f"    {key:<20} {shown}")


def run_single(model_key: str, strategy_key: str) -> None:
    """Run one condition and merge it into the manifest."""
    print("=" * 92)
    print(f"04_grid.py - SINGLE CONDITION: {model_key} / {strategy_key}")
    print("=" * 92)

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SAMPLE_PATH}\nRun `python src/01_prepare.py` first.")

    df = pd.read_parquet(SAMPLE_PATH)
    X = make_features(df)
    y = df["is_fraud"].to_numpy()
    amounts = df["amt"].to_numpy()
    n_outer = N_SPLITS_OUTER * N_REPEATS_OUTER
    fits = n_outer * (N_SPLITS_INNER * N_ITER_SEARCH + 1)

    print(f"\n  rows      : {len(df):,}   positives {y.sum():,} "
          f"({y.mean():.3%})")
    print(f"  outer     : {n_outer} folds")
    print(f"  inner     : StratifiedKFold({N_SPLITS_INNER}) + "
          f"RandomizedSearchCV(n_iter={N_ITER_SEARCH}) on {SEARCH_SCORING}")
    print(f"  => {fits} fits")
    print(f"  strategy  : {STRATEGIES[strategy_key]['note']}")
    print(f"\n  Appending to {MANIFEST_PATH.name}; other conditions untouched.")
    print(flush=True)

    t0 = time.time()
    rows = run_condition(model_key, strategy_key, X, y, amounts)
    wall = time.time() - t0

    s = summarise(model_key, strategy_key, rows)
    secs = np.array([r["seconds"] for r in rows])
    print(f"\n  WALL CLOCK : {wall:.1f}s = {wall / 60:.1f} min")
    print(f"  per fold   : mean {secs.mean():.1f}s  min {secs.min():.1f}s  "
          f"max {secs.max():.1f}s")
    print(f"\n  AUPRC      : {s['pr_auc_mean']:.4f} (sd {s['pr_auc_std']:.4f})")
    print(f"  ROC-AUC    : {s['roc_auc_mean']:.4f}")
    print(f"  F1@0.5     : {s['f1_at_0.5_mean']:.4f}  "
          f"prec {s['precision_at_0.5_mean']:.1%}  "
          f"recall {s['recall_at_0.5_mean']:.1%}")
    print(f"\n  selected hyperparameters across {len(rows)} folds:")
    _show_selected(rows)

    replaced, n_s, n_f = append_to_manifest(rows, s)
    if replaced:
        print(f"\n  replaced {replaced} existing rows for this condition")
    print(f"\n  {MANIFEST_PATH.name}: {n_s} conditions, {n_f} folds")
    print("=" * 92)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the model x strategy grid, or a single condition.")
    parser.add_argument("--model", choices=list(MODELS),
                        help="run only this model (requires --strategy)")
    parser.add_argument("--strategy", choices=list(STRATEGIES),
                        help="run only this strategy (requires --model)")
    args = parser.parse_args()

    if bool(args.model) != bool(args.strategy):
        parser.error("--model and --strategy must be given together")

    # One condition: run it and APPEND. The full loop below overwrites the
    # manifest, which would discard every condition not in this run.
    if args.model:
        run_single(args.model, args.strategy)
        return

    print("=" * 92)
    print("04_grid.py - model x imbalance strategy, nested CV within each")
    print("=" * 92)

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SAMPLE_PATH}\nRun `python src/01_prepare.py` first.")

    df = pd.read_parquet(SAMPLE_PATH)
    X = make_features(df)
    y = df["is_fraud"].to_numpy()
    amounts = df["amt"].to_numpy()

    conditions = [(m, s) for m in MODELS for s in STRATEGIES]
    n_outer = N_SPLITS_OUTER * N_REPEATS_OUTER

    print(f"\n  rows            : {len(df):,}   positives {y.sum():,} "
          f"({y.mean():.3%})")
    print(f"  models          : {', '.join(MODELS)}")
    print(f"  strategies      : {', '.join(STRATEGIES)}")
    print(f"  conditions      : {len(conditions)} "
          f"({len(MODELS)} models x {len(STRATEGIES)} strategies)")
    print(f"  outer           : RepeatedStratifiedKFold("
          f"{N_SPLITS_OUTER}x{N_REPEATS_OUTER}) = {n_outer} folds per condition")
    print(f"  inner           : StratifiedKFold({N_SPLITS_INNER}) + "
          f"RandomizedSearchCV(n_iter={N_ITER_SEARCH}), scoring "
          f"{SEARCH_SCORING}")
    print(f"  total inner fits: "
          f"{len(conditions) * n_outer * N_ITER_SEARCH * N_SPLITS_INNER:,}")
    print(f"\n  The imbalance strategy is FIXED within each condition and is")
    print(f"  never a search parameter. Only model hyperparameters are tuned.")
    for key, strat in STRATEGIES.items():
        print(f"    {key:<13} {strat['note']}")

    print_search_spaces()

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for n, (model_key, strategy_key) in enumerate(conditions, 1):
        label = f"{model_key} / {strategy_key}"
        print(f"\n[{n}/{len(conditions)}] {label}", flush=True)
        t0 = time.time()
        rows = run_condition(model_key, strategy_key, X, y, amounts)
        all_rows.extend(rows)
        pr = np.array([r["pr_auc"] for r in rows])
        roc = np.array([r["roc_auc"] for r in rows])
        f1 = np.array([r["f1_at_0.5"] for r in rows])
        print(f"      AUPRC {pr.mean():.4f} (sd {pr.std(ddof=1):.4f})   "
              f"ROC-AUC {roc.mean():.4f}   F1@.5 {f1.mean():.4f}   "
              f"[{time.time() - t0:.0f}s]", flush=True)

    # ---- Condition comparison -------------------------------------------
    print("\n" + "=" * 92)
    print("CONDITIONS RANKED BY AUPRC  (mean over 15 outer folds)")
    print("=" * 92)
    print(f"{'model':>8} {'strategy':>13} {'AUPRC':>8} {'sd':>7} "
          f"{'ROC-AUC':>9} {'F1@.5':>8} {'prec@.5':>9} {'recall@.5':>10}")
    print("-" * 92)

    summaries = []
    for model_key, strategy_key in conditions:
        rows = [r for r in all_rows
                if r["model"] == model_key and r["strategy"] == strategy_key]
        summaries.append(summarise(model_key, strategy_key, rows))

    for s in sorted(summaries, key=lambda d: -d["pr_auc_mean"]):
        print(f"{s['model']:>8} {s['strategy']:>13} {s['pr_auc_mean']:>8.4f} "
              f"{s['pr_auc_std']:>7.4f} {s['roc_auc_mean']:>9.4f} "
              f"{s['f1_at_0.5_mean']:>8.4f} "
              f"{s['precision_at_0.5_mean']:>9.1%} "
              f"{s['recall_at_0.5_mean']:>10.1%}")
    print("-" * 92)

    best = max(summaries, key=lambda d: d["pr_auc_mean"])
    worst = min(summaries, key=lambda d: d["pr_auc_mean"])
    print(f"\n  best  : {best['model']} / {best['strategy']}  "
          f"AUPRC {best['pr_auc_mean']:.4f}")
    print(f"  worst : {worst['model']} / {worst['strategy']}  "
          f"AUPRC {worst['pr_auc_mean']:.4f}")
    print(f"  spread: {best['pr_auc_mean'] - worst['pr_auc_mean']:.4f}, against")
    print(f"          a typical within-condition sd of "
          f"{np.mean([s['pr_auc_std'] for s in summaries]):.4f}")
    print(f"\n  Compare any two conditions against that sd before calling one")
    print(f"  better. Differences smaller than it are fold noise.")

    # ---- Effect of strategy, holding model fixed -------------------------
    print("\n" + "=" * 92)
    print("EFFECT OF IMBALANCE STRATEGY, WITHIN EACH MODEL")
    print("=" * 92)
    print(f"  AUPRC change versus that model's 'none' condition.\n")
    for model_key in MODELS:
        base = next(s["pr_auc_mean"] for s in summaries
                    if s["model"] == model_key and s["strategy"] == "none")
        print(f"  {model_key}  (none = {base:.4f})")
        for strategy_key in STRATEGIES:
            if strategy_key == "none":
                continue
            v = next(s["pr_auc_mean"] for s in summaries
                     if s["model"] == model_key
                     and s["strategy"] == strategy_key)
            sd = next(s["pr_auc_std"] for s in summaries
                      if s["model"] == model_key
                      and s["strategy"] == strategy_key)
            verdict = "within noise" if abs(v - base) < sd else (
                "BETTER" if v > base else "WORSE")
            print(f"    {strategy_key:<13} {v:.4f}  {v - base:+.4f}  {verdict}")
        print()

    # ---- Selected hyperparameters ----------------------------------------
    print("=" * 92)
    print("SELECTED HYPERPARAMETERS, BY CONDITION")
    print("=" * 92)
    for model_key, strategy_key in conditions:
        rows = [r for r in all_rows
                if r["model"] == model_key and r["strategy"] == strategy_key]
        print(f"\n  {model_key} / {strategy_key}  ({len(rows)} folds)")
        keys = sorted({k for r in rows for k in r["best_params"]})
        for key in keys:
            vals = [r["best_params"][key] for r in rows]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in vals):
                arr = np.array(vals, dtype=float)
                if arr.min() > 0 and arr.max() / max(arr.min(), 1e-12) > 20:
                    centre = f"geo-mean {np.exp(np.mean(np.log(arr))):.4g}"
                else:
                    centre = f"median {np.median(arr):.4g}"
                print(f"    {key:<22} {centre:<22} "
                      f"range {arr.min():.4g} to {arr.max():.4g}")
            else:
                counts = {}
                for v in vals:
                    counts[str(v)] = counts.get(str(v), 0) + 1
                top = sorted(counts.items(), key=lambda kv: -kv[1])
                shown = "  ".join(f"{k} x{n}" for k, n in top[:4])
                print(f"    {key:<22} {shown}")

    print(f"\n  A parameter whose selected value swings across orders of")
    print(f"  magnitude is not a finding - the inner scores are flat in it and")
    print(f"  the search is picking noise. Report the procedure, not 'the'")
    print(f"  chosen value.")

    MANIFEST_PATH.write_text(json.dumps({
        **manifest_metadata(),
        "summaries": summaries,
        "folds": all_rows,
    }, indent=2))

    n_files = len(list(PRED_DIR.glob("*__*_r*_f*.npz")))
    total_mb = sum(f.stat().st_size
                   for f in PRED_DIR.glob("*__*_r*_f*.npz")) / 1e6
    print(f"\n  wrote {n_files} .npz files ({total_mb:.1f} MB) named")
    print(f"    <model>__<strategy>_r<repeat>_f<fold>.npz")
    print(f"  wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 92)


if __name__ == "__main__":
    main()
