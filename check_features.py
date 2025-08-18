import joblib
import pandas as pd

# Load the feature columns
feature_columns = joblib.load("models/feature_columns.pkl")
print("Expected feature columns:")
for i, col in enumerate(feature_columns):
    print(f"{i+1:2d}. {col}")

print(f"\nTotal features: {len(feature_columns)}")

# Also check what's missing
print("\nAll feature columns:")
for i, col in enumerate(feature_columns):
    print(f"{col}")
