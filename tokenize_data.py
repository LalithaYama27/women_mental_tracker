import pandas as pd
from transformers import BertTokenizer
import torch

# Load the dataset
dataset = pd.read_csv(r"C:\Users\welcome\Downloads\women_mental_tracker\cleaned_Local Stories on Maternal Sexual and Reproductive Health in Africa.csv")  # Replace with the path to your dataset

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the tokenization function
def tokenize_function(examples):
    return tokenizer(examples['Story'], padding=True, truncation=True, max_length=512)

# Apply the tokenization function to your dataset's 'Story' column
tokenized_input = dataset['Story'].apply(lambda x: tokenize_function({'Story': x}))

# You can convert the tokenized input to a PyTorch tensor if needed
input_ids = tokenized_input.apply(lambda x: torch.tensor(x['input_ids']))
attention_mask = tokenized_input.apply(lambda x: torch.tensor(x['attention_mask']))

# You can now add these tensors to your dataset
dataset['input_ids'] = input_ids
dataset['attention_mask'] = attention_mask

# Save the tokenized dataset if you need to
dataset.to_csv('tokenized_dataset.csv', index=False)

# Check the tokenized data
print(dataset[['Story', 'input_ids', 'attention_mask']].head())
