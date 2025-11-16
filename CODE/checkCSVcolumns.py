import pandas as pd

csv_path = "../archive/imgs_2023071012133740345.csv"
df = pd.read_csv(csv_path, nrows=5)

print("=" * 80)
print("CSV COLUMNS ANALYSIS")
print("=" * 80)
print(f"\nTotal columns: {len(df.columns)}")
print("\nAll columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "=" * 80)
print("CHECKING FOR CATEGORICAL FEATURES")
print("=" * 80)
required_cats = ['RUType', 'Mode', 'load_bin']
for cat in required_cats:
    if cat in df.columns:
        print(f"✓ {cat}: FOUND")
        print(f"  Sample values: {df[cat].head().tolist()}")
    else:
        print(f"✗ {cat}: MISSING")

print("\n" + "=" * 80)
print("SAMPLE DATA (first 2 rows):")
print("=" * 80)
print(df.head(2).T)