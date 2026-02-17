import pandas as pd

INPUT_FILE = 'normalized_landmarks.csv'
OUTPUT_FILE = 'balanced_landmarks.csv'

TARGET_COUNT = 300

# 1. Load the dataset
df = pd.read_csv(INPUT_FILE)

for i in df["label"].unique():

    
    TARGET_CLASS=i
    # 2. Separate the target class from the other classes

    target_df = df[df['label'] == TARGET_CLASS]
    other_df = df[df['label'] != TARGET_CLASS]

    print(f"Original count for '{TARGET_CLASS}': {len(target_df)}")

    # 3. Downsample the target class
    # We use random_state=42 for reproducibility
    if len(target_df) > TARGET_COUNT:
        target_df_reduced = target_df.sample(n=TARGET_COUNT, random_state=42)
        print(f"Reduced '{TARGET_CLASS}' to {TARGET_COUNT} samples.")
    else:
        target_df_reduced = target_df
        print(f"'{TARGET_CLASS}' already has fewer than {TARGET_COUNT} samples. No reduction needed.")

    # 4. Combine the reduced class back with the other data
    balanced_df = pd.concat([target_df_reduced, other_df])
    df=balanced_df

# 5. Shuffle the dataset
# It's good practice to shuffle so the ANN doesn't see all of one class at once
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Save the result
balanced_df.to_csv(OUTPUT_FILE, index=False)
print(f"Balanced dataset saved as {OUTPUT_FILE}. Total rows: {len(balanced_df)}")
print(balanced_df)