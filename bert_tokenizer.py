from transformers import BertTokenizer
import pandas as pd

def tokenize_data(data_path):
    # Load cleaned data
    dataset = pd.read_csv(data_path)

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the 'Story' column
    encodings = tokenizer(
        dataset['Story'].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    return encodings, dataset['target_text'].tolist()
