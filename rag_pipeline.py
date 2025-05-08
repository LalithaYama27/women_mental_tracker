import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd

# Load tokenizer and model (use distilbert-base-uncased for 768-dimensional embeddings)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Function to get embedding for query
def get_embedding(text):
    model.eval()
    with torch.no_grad():
        encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        output = model(**encoded_input)
        embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding.astype("float32")

# Load FAISS index
index = faiss.read_index("faiss_index.bin")

# Load original data
df = pd.read_csv("train_data.csv")
text_column = 'input_text'

# Check FAISS index dimensionality
index_dimension = index.d
print(f"FAISS index dimensionality: {index_dimension}")

# Get user query
query = input("Enter your query: ")

# Get embedding for the query (must match FAISS index dimension)
query_vector = get_embedding(query).reshape(1, -1)

# Ensure the query's embedding dimension matches FAISS index dimension
if query_vector.shape[1] != index_dimension:
    raise ValueError(f"Query embedding dimension {query_vector.shape[1]} does not match FAISS index dimension {index_dimension}.")

# Perform similarity search
k = 5  # number of similar results
distances, indices = index.search(query_vector, k)

# Display results
print("\nTop matching results:")
for idx, score in zip(indices[0], distances[0]):
    print(f"\nScore: {score:.4f}")
    print(f"Text: {df.iloc[idx][text_column]}")
