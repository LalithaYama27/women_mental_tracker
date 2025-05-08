import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import pickle  # Import pickle for saving the id_map.pkl file

# Load your dataset
df = pd.read_csv("C:/Users/welcome/Downloads/women_mental_tracker/train_data.csv")

# Load pre-trained transformer model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings using the transformer model
def generate_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean of the hidden states (pooling) to get a fixed-size embedding
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()  # shape: (num_texts, embedding_size)
    return embeddings

# Convert the 'input_text' column to embeddings
embeddings = generate_embeddings(df['input_text'].tolist())

# Create a FAISS index (FlatL2 in this case for simplicity)
dimension = embeddings.shape[1]  # Get the embedding dimension
index = faiss.IndexFlatL2(dimension)

# Add the embeddings to the index
index.add(embeddings)

# Save the FAISS index
faiss.write_index(index, "faiss_index.bin")

# Create and save the id_map.pkl file
id_map = {i: text for i, text in enumerate(df['input_text'])}  # Mapping index to text
with open('id_map.pkl', 'wb') as f:
    pickle.dump(id_map, f)

print("FAISS index rebuilt and saved as 'faiss_index.bin'.")
print("ID mapping saved as 'id_map.pkl'.")
