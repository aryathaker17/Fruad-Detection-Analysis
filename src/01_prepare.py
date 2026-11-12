"""
01_prepare.py - FIN 611 fraud detection research project

WHAT THIS SCRIPT DOES
  1. Loads fraudTrain.csv + fraudTest.csv from data/raw/
  2. Runs diagnostic checks on the `merchant` column (the 'fraud_' prefix and
     embedded commas) - both are traps in this dataset, explained below
  3. Parses trans_date_trans_time into a real timestamp
  4. Drops direct personal identifiers that carry no modeling signal
  5. Takes a stratified subsample of 400,000 rows that PRESERVES the
     original fraud rate CASE
  6. Saves data/processed/sample.parquet and prints what it did

Run from anywhere:  python src/01_prepare.py

Never modifies anything in data/raw/ - it is opened read-only.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Paths are resolved relative to THIS FILE, not the shell's working directory,
# so the script behaves identically no matter where you launch it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "sample.parquet"
META_PATH = PROCESSED_DIR / "sample_metadata.json"

TARGET_ROWS = 400_000

# A fixed seed makes the subsample reproducible: re-running this script gives
# the byte-identical sample, so every downstream result is traceable to it.
# Change it only if you intend to regenerate everything downstream.
RANDOM_SEED = 611

# Columns that identify a *person* rather than describing a *transaction*.
# These are dropped because they cannot generalize: a model that learns
# "transactions by Jennifer Banks are safe" has memorized a name, not fraud.
DROP_DIRECT_ID = [
    "first",      # cardholder first name
    "last",       # cardholder last name
    "street",     # cardholder street address
    "trans_num",  # per-transaction hash - unique per row, so pure noise
]

# dob is personally identifying, but cardholder AGE at transaction time is a
# legitimate and commonly useful feature. We drop dob here to honor the
# "no personal identifiers" rule. If you later want age, derive it in
# 02_explore.py or a feature script from the raw file:
#     age = (trans_date_trans_time - pd.to_datetime(dob)).dt.days // 365
DROP_PII_DATE = ["dob"]

# Kept deliberately, though you may still choose to drop some later. These are
# NOT direct identifiers - each one is a plausible model input:
#   cc_num ....... card identifier; needed to build per-card history features
#                  (and listed as a key column in instructions.md). Do NOT
#                  feed it to a model as a raw number - it is an ID, not a
#                  quantity. It is here so you can GROUP BY it.
#   gender, job .. cardholder demographics
#   city, state, zip, lat, long, city_pop ..... where the cardholder lives
#   merch_lat, merch_long .... where the merchant is; the distance between
#                  cardholder and merchant is a classic fraud feature
#   unix_time .... same instant as trans_date_trans_time, as an integer;
#                  convenient for time-difference features
KEPT_FOR_REVIEW = [
    "cc_num", "gender", "job", "city", "state", "zip",
    "lat", "long", "city_pop", "merch_lat", "merch_long", "unix_time",
]


def load_raw() -> pd.DataFrame:
    """Read both CSVs and stack them into one frame."""
    frames = []
    for name in ("fraudTrain.csv", "fraudTest.csv"):
        path = RAW_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}\n"
                "Download it from "
                "https://www.kaggle.com/datasets/kartik2112/fraud-detection "
                "and place both CSVs in data/raw/ (see instructions.md)."
            )

        print(f"  reading {name} ...", flush=True)
        df = pd.read_csv(
            path,
            # The first column is an unnamed row counter. index_col=0 parks it
            # in the index so it never becomes a feature; we discard it below.
            index_col=0,
            # Parse the timestamp at read time rather than converting after.
            parse_dates=["trans_date_trans_time"],
            # cc_num is read as TEXT on purpose. It is a 16-digit identifier;
            # as an integer it invites meaningless arithmetic (a "mean card
            # number" is nonsense) and risks precision loss in some tools.
            dtype={"cc_num": "string"},
        )

        # Record which file each row came from. fraudTrain and fraudTest are
        # split by TIME, not at random - test covers the later period. Once the
        # two are stacked that boundary is invisible, so we preserve it. See
        # the note printed at the end of this script.
        df["source"] = name.replace("fraud", "").replace(".csv", "").lower()

        frames.append(df.reset_index(drop=True))

    return pd.concat(frames, ignore_index=True)


def check_merchant_column(df: pd.DataFrame) -> None:
    """
    Diagnostics on `merchant`. Prints findings; changes nothing.

    Two things to check, both of which mislead people on this dataset.
    """
    print("\n" + "-" * 68)
    print("MERCHANT COLUMN CHECKS")
    print("-" * 68)

    merchant = df["merchant"]
    n_rows = len(df)

    # --- Check 1: the 'fraud_' prefix -------------------------------------
    # Every merchant name in this dataset begins with 'fraud_' - it is an
    # artifact of the Sparkov data generator, which prefixed its whole
    # merchant table. It is NOT a fraud label.
    #
    # Why this matters: if the prefix appeared only on fraudulent rows, using
    # merchant as a feature would leak the answer and your model would score
    # ~100% while learning nothing. We verify it appears at the SAME rate in
    # both classes, which proves it carries no information about is_fraud.
    has_prefix = merchant.str.startswith("fraud_")
    print(f"  rows with 'fraud_' prefix : {has_prefix.sum():,} / {n_rows:,} "
          f"({has_prefix.mean():.2%})")

    prefix_by_class = df.groupby("is_fraud")["merchant"].apply(
        lambda s: s.str.startswith("fraud_").mean()
    )
    for label, rate in prefix_by_class.items():
        kind = "fraud" if label == 1 else "legitimate"
        print(f"    among {kind:11} rows : {rate:.2%}")

    if has_prefix.all():
        print("  -> prefix is on 100% of rows in BOTH classes: it is a naming")
        print("     artifact, not a label. Safe to strip; nothing is lost.")
    else:
        # If this ever triggers, stop and investigate before modeling.
        print("  -> WARNING: prefix is NOT universal. If it is more common in")
        print("     one class, `merchant` leaks the target. Investigate before")
        print("     using merchant as a feature.")

    # --- Check 2: embedded commas ----------------------------------------
    # Many merchant names contain commas, e.g. "fraud_Rippin, Kub and Mann".
    # The CSV quotes those fields, so a proper parser handles them - but a
    # naive line.split(',') would shift every later column, silently
    # corrupting the data. We confirm the parse was clean.
    has_comma = merchant.str.contains(",", regex=False)
    print(f"\n  merchant values containing a comma : {has_comma.sum():,} rows"
          f" ({has_comma.mean():.2%})")
    print(f"    distinct such merchants : "
          f"{merchant[has_comma].nunique():,} of {merchant.nunique():,}")

    # Parse-integrity check: these columns are positioned AFTER merchant, so
    # if quoting had been mishandled they would be full of garbage or nulls.
    tail_cols = ["amt", "is_fraud"]
    nulls = df[tail_cols].isna().sum()
    print(f"    nulls in columns after merchant : "
          f"{dict(zip(tail_cols, nulls))}")
    if nulls.sum() == 0 and df["amt"].dtype.kind == "f":
        print("  -> quoted commas parsed correctly: `amt` is numeric and")
        print("     columns after `merchant` are intact. (`job` is quoted the")
        print("     same way and is also fine.)")
    else:
        print("  -> WARNING: columns after `merchant` look damaged. The quoted")
        print("     commas may have been mis-parsed.")

    print("-" * 68)


def main() -> None:
    print("=" * 68)
    print("01_prepare.py - building the modeling sample")
    print("=" * 68)

    # ---- 1. Load ---------------------------------------------------------
    print("\n[1/5] Loading raw CSVs (read-only)")
    df = load_raw()
    n_raw = len(df)
    print(f"  loaded {n_raw:,} rows x {df.shape[1]} columns")
    print(f"  timestamp dtype: {df['trans_date_trans_time'].dtype}")
    print(f"  date range     : {df['trans_date_trans_time'].min()}"
          f"  ->  {df['trans_date_trans_time'].max()}")

    # ---- 2. Merchant diagnostics ----------------------------------------
    check_merchant_column(df)

    # The prefix is pure noise, so strip it for readability. This is cosmetic:
    # it removes 6 identical characters from every value and changes no
    # relationship in the data.
    df["merchant"] = df["merchant"].str.removeprefix("fraud_")
    print(f"\n[2/5] Stripped 'fraud_' prefix from merchant names")
    print(f"  example: {df['merchant'].iloc[0]!r}")

    # ---- 3. Drop personal identifiers -----------------------------------
    print(f"\n[3/5] Dropping personal identifier columns")
    to_drop = [c for c in DROP_DIRECT_ID + DROP_PII_DATE if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"  dropped ({len(to_drop)}): {', '.join(to_drop)}")
    print(f"  kept, but review before modeling: {', '.join(KEPT_FOR_REVIEW)}")
    print(f"  -> see the comments at the top of this file for why each was kept")
    print(f"  {df.shape[1]} columns remain")

    # ---- 4. Stratified subsample ----------------------------------------
    print(f"\n[4/5] Stratified subsample to {TARGET_ROWS:,} rows")

    pop_rate = df["is_fraud"].mean()
    print(f"  population fraud rate : {pop_rate:.4%} "
          f"({df['is_fraud'].sum():,} of {len(df):,})")

    # train_test_split with stratify=is_fraud draws a simple random sample that
    # holds the class proportions fixed. Every row - fraud or not - has the
    # same chance of selection, so the result is a MINIATURE of the full
    # dataset rather than a reweighted version of it.
    #
    # The consequence worth understanding: this keeps only about
    # TARGET_ROWS * pop_rate frauds, discarding most of the minority class.
    # What you buy for that is interpretability. Because the sample is
    # representative, precision, PR-AUC and predicted probabilities all mean
    # what they say, and no downstream script needs a correction factor.
    #
    # We only need the first piece; the remainder is discarded.
    sample, _ = train_test_split(
        df,
        train_size=TARGET_ROWS,
        stratify=df["is_fraud"],
        random_state=RANDOM_SEED,
        shuffle=True,          # also leaves the file in random order, so a
    )                          # head() peek is not an all-fraud slice
    sample = sample.reset_index(drop=True)

    sample_rate = sample["is_fraud"].mean()
    print(f"  sample fraud rate     : {sample_rate:.4%} "
          f"({sample['is_fraud'].sum():,} of {len(sample):,})")
    print(f"  difference            : {abs(sample_rate - pop_rate):.6%} "
          f"({'MATCH' if abs(sample_rate - pop_rate) < 1e-4 else 'CHECK THIS'})")
    print(f"\n  Note: stratification is on is_fraud only. Category and source")
    print(f"  proportions are preserved by the randomness of the draw rather")
    print(f"  than enforced exactly; at {TARGET_ROWS:,} rows the drift is tiny.")

    # ---- 5. Save ---------------------------------------------------------
    print(f"\n[5/5] Saving")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Parquet keeps dtypes (timestamps stay timestamps, cc_num stays text) and
    # is far smaller and faster than CSV. Requires the `pyarrow` package.
    sample.to_parquet(OUT_PATH, index=False)
    print(f"  wrote {OUT_PATH.relative_to(PROJECT_ROOT)} "
          f"({OUT_PATH.stat().st_size / 1e6:.1f} MB)")

    # The sample is representative, so no class reweighting is needed
    # anywhere downstream and no scale factor is recorded: 04_cost.py reports
    # costs per 100,000 transactions, which needs no dataset-level constant.
    n_fraud = int(sample["is_fraud"].sum())
    metadata = {
        "random_seed": RANDOM_SEED,
        "sampling_method": "train_test_split, stratified on is_fraud",
        "rows": int(len(sample)),
        "frauds": n_fraud,
        "sample_fraud_rate": n_fraud / len(sample),
        "population_rows": int(n_raw),
        "population_frauds": int(df["is_fraud"].sum()),
        "population_fraud_rate": float(df["is_fraud"].mean()),
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"  wrote {META_PATH.relative_to(PROJECT_ROOT)}")

    # ---- Report ----------------------------------------------------------
    print("\n" + "=" * 68)
    print("RESULT")
    print("=" * 68)
    print(f"  shape        : {sample.shape[0]:,} rows x {sample.shape[1]} columns")
    print(f"  fraud count  : {n_fraud:,}")
    print(f"  fraud rate   : {n_fraud / len(sample):.3%}  (in this sample)")
    print(f"  columns      : {', '.join(sample.columns)}")

    print(f"\n  CLASS BALANCE PRESERVED")
    print(f"  This sample is a miniature of the full dataset: the fraud rate")
    print(f"  is {n_fraud / len(sample):.4%} against a population "
          f"{df['is_fraud'].mean():.4%}.")
    print(f"  Consequences, all of them good for interpretation:")
    print(f"    * Precision, PR-AUC and F1 mean what they say. No correction.")
    print(f"    * Predicted probabilities are calibrated to reality - a model")
    print(f"      output of 0.30 really is about a 30% chance of fraud.")
    print(f"    * Cost analysis needs no per-class weights, and costs scale")
    print(f"      linearly with transaction count.")
    print(f"\n  THE PRICE YOU PAID")
    print(f"  Only {n_fraud:,} frauds remain, down from "
          f"{int(df['is_fraud'].sum()):,} in the full data")
    print(f"  - {1 - n_fraud / df['is_fraud'].sum():.1%} of the minority class was discarded.")
    print(f"  With 5-fold CV that is ~{n_fraud // 5:,} frauds per test fold, so")
    print(f"  fraud-dependent metrics will be noisier than before. Expect")
    print(f"  wider fold-to-fold spread and judge improvements against it.")
    print("\n  Also note: `source` marks which file each row came from.")
    print("  fraudTrain/fraudTest are split by TIME, so for an honest")
    print("  evaluation prefer a time-based split over a random one.")
    print("=" * 68)


if __name__ == "__main__":
    main()
