import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to get embedding for query
def get_embedding(text):
    model.eval()
    with torch.no_grad():
        encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        output = model(**encoded_input)
        embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding.astype("float32")

# Load FAISS index
index = faiss.read_index("faiss_index.index")

# Load original data
df = pd.read_csv("train_data.csv")
text_column = 'input_text'

# Get user query
query = input("Enter your query: ")

# Get embedding for the query
query_vector = get_embedding(query).reshape(1, -1)

# Perform similarity search
k = 5  # number of similar results
distances, indices = index.search(query_vector, k)

# Display results
print("\nTop matching results:")
for idx, score in zip(indices[0], distances[0]):
    print(f"\nScore: {score:.4f}")
    print(f"Text: {df.iloc[idx][text_column]}")