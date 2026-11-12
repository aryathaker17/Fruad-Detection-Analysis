"""
03_baseline.py - FIN 611 fraud detection research project

THE REFERENCE POINT
  Plain logistic regression, untuned, with NO imbalance handling at all:
    * no class_weight="balanced"
    * no SMOTE, no over- or under-sampling
    * no hyperparameter search - scikit-learn defaults apart from max_iter
    * no threshold tuning
  Nothing here is chosen by looking at results, so there is nothing to tune
  honestly or dishonestly. That is the point: this is the number every
  condition in 04_grid.py has to beat, and it is cheap and unambiguous.

  Read it alongside 04_grid.py, which tunes model hyperparameters by nested
  cross-validation under each imbalance strategy as a separate condition.

VALIDATION
  RepeatedStratifiedKFold, 5 splits x 3 repeats = 15 fits. Stratified keeps
  the fraud rate identical in every fold; repeating the whole split three
  times shows how much a score moves purely because of where the fold
  boundaries fell.

  There is no inner loop because there is nothing to select. The scores below
  are honest as they stand.

Run from anywhere:  python src/03_baseline.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "sample.parquet"
PRED_DIR = PROJECT_ROOT / "results" / "predictions"
MANIFEST_PATH = PRED_DIR / "baseline_logreg_manifest.json"

MODEL_NAME = "baseline_logreg"
N_SPLITS = 5
N_REPEATS = 3
RANDOM_SEED = 611
TOP_COEFFICIENTS = 10

NUMERIC_FEATURES = [
    "amt",            # raw amount - the strongest single signal
    "log_amt",        # amounts are heavily right-skewed; the log lets a linear
                      # model express "risk rises with order of magnitude"
    "hour_sin",       # time of day on a circle - see make_features()
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "distance_km",    # cardholder home to merchant
    "log_city_pop",
]
CATEGORICAL_FEATURES = ["category", "gender"]

# Deliberately excluded:
#   cc_num, merchant, city, job, zip ... 497-999 distinct values each; one-hot
#       would add hundreds of columns and invite memorising individual cards.
#   state ............. 51 values, weak signal, 50 extra columns.
#   lat, long ......... constant per card here, so effectively a cardholder ID.
#                       distance_km carries the geography without that problem.
#   unix_time ......... monotonic; effectively a row counter.
#   source ............ metadata about data assembly, not a transaction property.


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in kilometres.

    Euclidean distance on raw lat/long is wrong: a degree of longitude is
    ~111 km at the equator and shrinks to nothing at the poles.
    """
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive model inputs from raw columns.

    Every transform here is ROW-WISE: one row's value depends only on that
    row, so computing them before splitting leaks nothing. Anything that
    learns from a COLLECTION of rows - a mean, a standard deviation, a
    category vocabulary - lives in the Pipeline instead and is refit on the
    training rows of every fold. Same principle as the project's rule that
    resampling happens inside training folds only.
    """
    out = pd.DataFrame(index=df.index)
    out["amt"] = df["amt"]
    out["log_amt"] = np.log1p(df["amt"])

    # Hour is CIRCULAR: 23:00 and 00:00 are one hour apart but 23 units apart
    # as plain numbers. The fraud spike runs continuously across midnight, so
    # mapping the hour onto a circle keeps those hours adjacent.
    hour = df["trans_date_trans_time"].dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dow = df["trans_date_trans_time"].dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    out["distance_km"] = haversine_km(
        df["lat"].to_numpy(), df["long"].to_numpy(),
        df["merch_lat"].to_numpy(), df["merch_long"].to_numpy(),
    )
    out["log_city_pop"] = np.log1p(df["city_pop"])
    out["category"] = df["category"]
    out["gender"] = df["gender"]
    return out


def build_pipeline() -> Pipeline:
    """
    Preprocessing + model as ONE object, so fit() learns the scaler's mean and
    the encoder's vocabulary from TRAINING rows only. Scaling the whole dataset
    once before the loop would let every fold's "unseen" test rows contribute
    to the numbers the model trained on.
    """
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"),
         CATEGORICAL_FEATURES),
    ])
    model = LogisticRegression(
        max_iter=1000,     # the default 100 does not converge on this data
        solver="lbfgs",
        class_weight=None,  # THE POINT: no help with the imbalance
        random_state=RANDOM_SEED,
    )
    return Pipeline([("prep", preprocess), ("clf", model)])


def main() -> None:
    print("=" * 76)
    print("03_baseline.py - untuned logistic regression, no imbalance handling")
    print("=" * 76)

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SAMPLE_PATH}\nRun `python src/01_prepare.py` first.")

    df = pd.read_parquet(SAMPLE_PATH)
    meta_path = SAMPLE_PATH.parent / "sample_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    X = make_features(df)
    y = df["is_fraud"].to_numpy()
    amounts = df["amt"].to_numpy()

    print(f"\n  rows               : {len(df):,}")
    print(f"  positives          : {y.sum():,} ({y.mean():.3%})")
    print(f"  model              : LogisticRegression(class_weight=None), "
          f"defaults")
    print(f"  imbalance handling : NONE")
    print(f"  hyperparameter tuning : NONE - nothing is selected from data")
    print(f"  cross-validation   : RepeatedStratifiedKFold("
          f"{N_SPLITS} x {N_REPEATS}) = {N_SPLITS * N_REPEATS} fits")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=RANDOM_SEED)

    print("\n" + "-" * 76)
    print(f"{'rep':>4} {'fold':>5} {'frauds':>7} {'ROC-AUC':>9} {'AUPRC':>8} "
          f"{'F1@.5':>8} {'prec@.5':>9} {'recall@.5':>10} {'secs':>6}")
    print("-" * 76)

    manifest = []
    roc_s, pr_s, f1_s, prec_s, rec_s = [], [], [], [], []
    coefs, feature_names = [], None

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        repeat, fold = i // N_SPLITS + 1, i % N_SPLITS + 1
        t0 = time.time()

        pipe = build_pipeline()
        pipe.fit(X.iloc[train_idx], y[train_idx])

        # Column 1 is P(fraud): predict_proba returns [P(class 0), P(class 1)].
        y_prob = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        y_true = y[test_idx]

        roc = roc_auc_score(y_true, y_prob)
        pr = average_precision_score(y_true, y_prob)
        flagged = y_prob >= 0.5
        f1 = f1_score(y_true, flagged.astype(int), zero_division=0)
        precision = (flagged & (y_true == 1)).sum() / max(flagged.sum(), 1)
        recall = (flagged & (y_true == 1)).sum() / max(y_true.sum(), 1)

        coefs.append(pipe.named_steps["clf"].coef_[0].copy())
        if feature_names is None:
            feature_names = pipe.named_steps["prep"].get_feature_names_out()

        out_path = PRED_DIR / f"{MODEL_NAME}_r{repeat}_f{fold}.npz"
        np.savez_compressed(
            out_path, y_true=y_true, y_prob=y_prob,
            amt=amounts[test_idx], test_index=test_idx,
            repeat=repeat, fold=fold,
        )

        elapsed = time.time() - t0
        roc_s.append(roc); pr_s.append(pr); f1_s.append(f1)
        prec_s.append(precision); rec_s.append(recall)
        print(f"{repeat:>4} {fold:>5} {y_true.sum():>7,} {roc:>9.4f} "
              f"{pr:>8.4f} {f1:>8.4f} {precision:>9.1%} {recall:>10.1%} "
              f"{elapsed:>6.1f}")

        manifest.append({
            "file": out_path.name, "repeat": repeat, "fold": fold,
            "n_test": int(len(test_idx)), "n_fraud": int(y_true.sum()),
            "roc_auc": float(roc), "pr_auc": float(pr),
            "f1_at_0.5": float(f1), "precision_at_0.5": float(precision),
            "recall_at_0.5": float(recall),
        })

    print("-" * 76)
    roc_a, pr_a, f1_a, prec_a, rec_a = map(
        np.array, (roc_s, pr_s, f1_s, prec_s, rec_s))
    print(f"{'mean':>18} {roc_a.mean():>9.4f} {pr_a.mean():>8.4f} "
          f"{f1_a.mean():>8.4f} {prec_a.mean():>9.1%} {rec_a.mean():>10.1%}")
    print(f"{'std dev':>18} {roc_a.std(ddof=1):>9.4f} {pr_a.std(ddof=1):>8.4f} "
          f"{f1_a.std(ddof=1):>8.4f} {prec_a.std(ddof=1):>9.1%} "
          f"{rec_a.std(ddof=1):>10.1%}")

    print(f"\n  AUPRC and PR-AUC are the same metric (average precision). Its")
    print(f"  no-skill baseline is the positive rate, {y.mean():.4f}, not 0.5 -")
    print(f"  so {pr_a.mean():.4f} is {pr_a.mean() / y.mean():.0f}x better than chance.")

    # ---- Coefficients ----------------------------------------------------
    coef_arr = np.vstack(coefs)
    mean_coef, sd_coef = coef_arr.mean(axis=0), coef_arr.std(axis=0, ddof=1)
    clean = [n.replace("num__", "").replace("cat__", "") for n in feature_names]
    top = np.argsort(-np.abs(mean_coef))[:TOP_COEFFICIENTS]

    print("\n" + "=" * 76)
    print(f"TOP {TOP_COEFFICIENTS} COEFFICIENTS BY MAGNITUDE")
    print("=" * 76)
    print(f"  Mean over the {len(coefs)} folds. Units are LOG-ODDS; the odds")
    print(f"  ratio is exp(coefficient).\n")
    print(f"{'#':>3} {'feature':<28} {'coef':>9} {'sd':>8} {'odds ratio':>11}")
    print("-" * 76)
    for rank, idx in enumerate(top, 1):
        print(f"{rank:>3} {clean[idx]:<28} {mean_coef[idx]:>9.4f} "
              f"{sd_coef[idx]:>8.4f} {np.exp(mean_coef[idx]):>11.3f}")
    print("-" * 76)
    unstable = [i for i in top if abs(mean_coef[i]) < 2 * sd_coef[i]]
    if unstable:
        print(f"\n  UNSTABLE ({len(unstable)}): "
              f"{', '.join(clean[i] for i in unstable)}")
    else:
        print(f"\n  All {TOP_COEFFICIENTS} have |mean| > 2 sd - signs are stable.")
    print(f"  Numeric features are standardized (effect per 1 sd); category")
    print(f"  dummies are 0/1 against the dropped reference level.")
    print(f"  amt and log_amt are collinear - read that pair together.")

    MANIFEST_PATH.write_text(json.dumps({
        "model": MODEL_NAME,
        "design": "untuned reference, no imbalance handling, no inner loop",
        "cv": f"RepeatedStratifiedKFold({N_SPLITS}x{N_REPEATS})",
        "random_seed": RANDOM_SEED,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "roc_auc_mean": float(roc_a.mean()), "roc_auc_std": float(roc_a.std(ddof=1)),
        "pr_auc_mean": float(pr_a.mean()), "pr_auc_std": float(pr_a.std(ddof=1)),
        "f1_at_0.5_mean": float(f1_a.mean()), "f1_at_0.5_std": float(f1_a.std(ddof=1)),
        "precision_at_0.5_mean": float(prec_a.mean()),
        "recall_at_0.5_mean": float(rec_a.mean()),
        "auprc_no_skill_baseline": float(y.mean()),
        "folds": manifest,
    }, indent=2))

    print(f"\n  wrote {len(manifest)} .npz files and the manifest to "
          f"{PRED_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"  each .npz holds: y_true, y_prob, amt, test_index, repeat, fold")
    print("=" * 76)


if __name__ == "__main__":
    main()
