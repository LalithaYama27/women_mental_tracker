import pandas as pd
from datasets import Dataset

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df[['input_text', 'target_text']]  # Adjust if your column names differ
    df = df.dropna()

    # Optional: convert labels to string if classification
    df['target_text'] = df['target_text'].astype(str)
    
    dataset = Dataset.from_pandas(df)
    return dataset
