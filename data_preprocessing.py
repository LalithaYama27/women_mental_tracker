import pandas as pd
from transformers import T5Tokenizer

# Initialize the tokenizer (you can replace it with the appropriate model you want)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Preprocess CSV data
def preprocess_csv_data(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    input_encodings = []
    target_encodings = []
    
    for index, row in df.iterrows():
        # Use the actual columns for input and target
        input_text = row['Scenario']  # 'Scenario' column as input text
        target_text = row['Guidelines Required']  # 'Guidelines Required' as target text
        
        # Encode the input and target text using the tokenizer
        input_encoding = tokenizer(input_text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        target_encoding = tokenizer(target_text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        
        input_encodings.append(input_encoding)
        target_encodings.append(target_encoding)
    
    return input_encodings, target_encodings

# Run preprocessing on your CSV
input_encodings, target_encodings = preprocess_csv_data("C:/Users/welcome/Downloads/women_mental_tracker/cleaned_General - MSRH.csv")

# Optionally, print the encoded inputs and targets
print("Encoded Input Examples:")
print(input_encodings[:2])  # Show the first 2 encoded examples
print("Encoded Target Examples:")
print(target_encodings[:2])  # Show the first 2 encoded examples
