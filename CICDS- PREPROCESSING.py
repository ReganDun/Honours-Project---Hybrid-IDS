import pandas as pd
import glob

# Path to your folder with all the CSVs
data_folder = "H:\\code\\MachineLearningCSV\\"  # or wherever your files are

# Get list of all CSVs (make sure only CICIDS CSVs are in this folder)
csv_files = glob.glob(data_folder + "*.csv")

# Empty list to hold each file's DataFrame
dataframes = []

print("Reading and combining CSV files...")

for file in csv_files:
    try:
        df = pd.read_csv(file, low_memory=False)
        dataframes.append(df)
        print(f"✅ Loaded {file}")
    except Exception as e:
        print(f"❌ Error loading {file}: {e}")

# Combine them
combined_df = pd.concat(dataframes, ignore_index=True)
print(f"\n✅ Combined Data Shape: {combined_df.shape}")

import numpy as np
from sklearn.model_selection import train_test_split

# Drop non-numeric / irrelevant columns
columns_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp',
                   'Source Port', 'Destination Port', 'Protocol']
combined_df = combined_df.drop(columns=columns_to_drop, errors='ignore')

# Drop NaNs and infinities
combined_df = combined_df.replace([np.inf, -np.inf], np.nan).dropna()

# Encode Label: BENIGN → 1, all else → 0
combined_df[' Label'] = combined_df[' Label'].apply(lambda x: 1 if x == 'BENIGN' else 0)

# Shuffle data (optional but recommended)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features/labels
X = combined_df.drop(' Label', axis=1)
y = combined_df[' Label']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Combine and save in your format (no headers)
train_data = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
test_data = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)

train_data.to_csv('H:\\code\\CICIDS2017_train_processedMONWED.csv', index=False, header=False)
test_data.to_csv('H:\\code\\CICIDS2017_test_processedMONWED.csv', index=False, header=False)

print("✅ Train and test data saved.")
