import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Load the dataset
df = pd.read_csv("train_data.csv")  # You can change the path if needed
text_column = 'input_text'  # Use the correct column name

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(texts):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts):
            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            output = model(**encoded_input)
            cls_embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
    return embeddings

# Generate embeddings
texts = df[text_column].tolist()
embeddings = get_embeddings(texts)

# Convert to numpy array
import numpy as np
embedding_matrix = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# Save the index
faiss.write_index(index, "faiss_index.index")

print("FAISS index created and saved successfully!")
