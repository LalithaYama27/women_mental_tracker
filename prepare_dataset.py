import pandas as pd
from datasets import Dataset

# Load your dataset
df = pd.read_csv(r"C:\Users\welcome\Downloads\women_mental_tracker\cleaned_General - MSRH.csv")

# Rename for consistency and clarity
df = df.rename(columns={
    'Scenario': 'input_text',
    'Midwifery / Medical solution ( advice)': 'target_text'
})

# Drop any rows with missing inputs or outputs
df = df[['input_text', 'target_text']].dropna()

# Convert to Hugging Face Dataset format
hf_dataset = Dataset.from_pandas(df)

# Split into train and validation sets
split_dataset = hf_dataset.train_test_split(test_size=0.2)

# Save locally to disk (optional)
split_dataset['train'].to_csv("train_data.csv", index=False)
split_dataset['test'].to_csv("val_data.csv", index=False)

print("Dataset prepared. Sample:")
print(split_dataset['train'][0])
