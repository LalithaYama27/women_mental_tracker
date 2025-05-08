import pandas as pd

# Use raw string to fix Windows path issue
df = pd.read_csv(r"C:\Users\welcome\Downloads\women_mental_tracker\cleaned_General - MSRH.csv")

# Show the first few rows and column names
print("Columns in the dataset:")
print(df.columns)
print("\nSample data:")
print(df.head())
