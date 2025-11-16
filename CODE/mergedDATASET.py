"""
Merge separate CSV files into complete training dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path("../archive")
OUTPUT_FILE = DATA_DIR / "merged_training_data.csv"

print("=" * 80)
print("MERGING 5G ENERGY CONSUMPTION DATASETS")
print("=" * 80)

# 1. Load BS metadata (static features)
print("\n1. Loading BS metadata...")
bs_meta = pd.read_csv(DATA_DIR / "imgs_2023071012123392536.csv")
print(f"   ✓ BS metadata: {bs_meta.shape}")
print(f"   Columns: {list(bs_meta.columns)}")

# 2. Load cell timeseries (dynamic features)
print("\n2. Loading cell timeseries...")
cell_ts = pd.read_csv(DATA_DIR / "imgs_2023071012130978799.csv")
print(f"   ✓ Cell timeseries: {cell_ts.shape}")
print(f"   Columns: {list(cell_ts.columns)}")

# 3. Load training targets
print("\n3. Loading training targets...")
targets = pd.read_csv(DATA_DIR / "imgs_2023071012133740345.csv")
print(f"   ✓ Targets: {targets.shape}")
print(f"   Columns: {list(targets.columns)}")

# 4. Merge datasets
print("\n4. Merging datasets...")

# First merge: targets + cell timeseries (on Time + BS)
print("   Step 1: Merging targets with cell timeseries...")
merged = targets.merge(
    cell_ts,
    on=['Time', 'BS'],
    how='left'
)
print(f"   ✓ After merge 1: {merged.shape}")

# Second merge: + BS metadata (on BS)
print("   Step 2: Merging with BS metadata...")
merged = merged.merge(
    bs_meta,
    on='BS',
    how='left',
    suffixes=('', '_meta')
)
print(f"   ✓ After merge 2: {merged.shape}")

# 5. Handle duplicates (CellName appears in both)
print("\n5. Cleaning columns...")
if 'CellName_meta' in merged.columns:
    # Keep cell_ts version, drop metadata version
    merged['CellName'] = merged['CellName'].fillna(merged['CellName_meta'])
    merged = merged.drop(columns=['CellName_meta'])

# 6. Check for missing values
print("\n6. Checking data quality...")
print(f"   Total rows: {len(merged)}")
print(f"   Complete cases: {merged.dropna().shape[0]}")
print(f"\n   Missing values per column:")
missing = merged.isnull().sum()
for col, count in missing[missing > 0].items():
    print(f"      {col}: {count} ({count/len(merged)*100:.1f}%)")

# 7. Fill missing values strategically
print("\n7. Filling missing values...")

# For ESMode columns (operational modes), fill with 0 (mode not active)
esmode_cols = [col for col in merged.columns if col.startswith('ESMode')]
for col in esmode_cols:
    merged[col] = merged[col].fillna(0)
    print(f"   ✓ {col}: filled with 0")

# For load, use forward fill within each BS group
if 'load' in merged.columns:
    merged['load'] = merged.groupby('BS')['load'].fillna(method='ffill').fillna(0)
    print(f"   ✓ load: forward-filled within BS groups")

# For categorical features, use mode
for col in ['RUType', 'Mode']:
    if col in merged.columns and merged[col].isnull().any():
        mode_val = merged[col].mode()[0] if len(merged[col].mode()) > 0 else 'Unknown'
        merged[col] = merged[col].fillna(mode_val)
        print(f"   ✓ {col}: filled with mode '{mode_val}'")

# For numeric features, use median
numeric_cols = merged.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if merged[col].isnull().any():
        median_val = merged[col].median()
        merged[col] = merged[col].fillna(median_val)
        print(f"   ✓ {col}: filled with median {median_val:.2f}")

# 8. Save merged dataset
print("\n8. Saving merged dataset...")
merged.to_csv(OUTPUT_FILE, index=False)
print(f"   ✓ Saved to: {OUTPUT_FILE}")

# 9. Final summary
print("\n" + "=" * 80)
print("MERGE COMPLETE!")
print("=" * 80)
print(f"\nFinal dataset shape: {merged.shape}")
print(f"Columns ({len(merged.columns)}):")
for i, col in enumerate(merged.columns, 1):
    dtype = merged[col].dtype
    unique = merged[col].nunique()
    print(f"   {i:2d}. {col:20s} ({dtype}, {unique} unique values)")

print("\n" + "=" * 80)
print("Sample data (first 2 rows):")
print("=" * 80)
print(merged.head(2).T)

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("1. Review the merged data above")
print("2. Update kerasTRAINER.py to use:")
print(f"   config.train_file = '{OUTPUT_FILE.name}'")
print("3. Run: python3 kerasTRAINER.py")
print("=" * 80)