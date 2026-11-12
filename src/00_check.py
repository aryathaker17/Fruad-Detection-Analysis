from pathlib import Path
import numpy as np
import pandas as pd

print("=== Files ===")
sample = Path("data/processed/sample.parquet")
print(f"sample.parquet: {'found' if sample.exists() else 'MISSING'}")

pred_dir = Path("results/predictions")
preds = sorted(pred_dir.glob("*.npz")) if pred_dir.exists() else []
print(f"prediction files: {len(preds)}")

if sample.exists():
    df = pd.read_parquet(sample)
    n_fraud = int(df["is_fraud"].sum())
    print("\n=== Data ===")
    print(f"rows: {len(df):,}")
    print(f"fraud: {n_fraud:,} ({n_fraud / len(df) * 100:.3f}%)")
    print(f"columns: {list(df.columns)}")
    print("\namount by class:")
    print(df.groupby("is_fraud")["amt"].describe()[["count", "mean", "50%", "max"]])

if preds:
    print("\n=== Predictions ===")
    d = np.load(preds[0])
    print(f"first file: {preds[0].name}")
    print(f"keys: {list(d.keys())}")
    for k in d.keys():
        print(f"  {k}: shape {d[k].shape}, dtype {d[k].dtype}")
    lengths = {k: len(d[k]) for k in d.keys() if d[k].ndim >= 1}
    print(f"lengths match: {len(set(lengths.values())) == 1}")