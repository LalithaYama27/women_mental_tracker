# Step 1: Import Libraries
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import torch
import os
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (if not already downloaded)
nltk.download('punkt')

# Step 2: Clean the 'Scenario' text
def clean_text(text):
    if isinstance(text, str):  # Check if the input is a string
        # Remove non-alphabetic characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Convert to lowercase
        text = text.lower()
        return text
    else:
        return ""  # Return an empty string if the input is not a valid string

# Step 3: Load the dataset and display basic information
folder_path = "C:/Users/welcome/Downloads/women_mental_tracker"
files = os.listdir(folder_path)

# Print the list of files to identify the dataset file
print(files)

# After identifying the dataset file, load it
file_path = "C:/Users/welcome/Downloads/women_mental_tracker/cleaned_General - MSRH.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Check for missing values in the dataset
print("Missing values per column:")
print(df.isnull().sum())

# Display the column names to check for irrelevant columns
print("\nColumns in the dataset:")
print(df.columns)

# Step 4: Tokenize the 'Scenario' column
df['tokenized_scenario'] = df['Scenario'].apply(lambda x: word_tokenize(x.lower()))  # Convert to lowercase for uniformity

# Show the first few tokenized scenarios
print("\nFirst few tokenized scenarios:")
print(df[['Scenario', 'tokenized_scenario']].head())

# Step 5: Clean the 'Scenario' column
df_cleaned = df.dropna(axis=1, how='all')  # Remove columns with all NaN values
df_cleaned = df_cleaned[['Scenario']]  # Keep only the 'Scenario' column

# Clean the text in the 'Scenario' column
df_cleaned['cleaned_scenario'] = df_cleaned['Scenario'].apply(clean_text)

# Check the cleaned data
print("\nCleaned Data (first 5 rows):")
print(df_cleaned.head())
