import pandas as pd
import os

# Define the new folder path where datasets are stored
folder_path = r"C:\Users\welcome\Downloads\women_mental_tracker"

# File paths (ensure filenames match exactly)
file_paths = {
    "General_MSRA": os.path.join(folder_path, "General - MSRH.csv"),
    "Local_Stories": os.path.join(folder_path, "Local Stories on Maternal Sexual and Reproductive Health in Africa.csv"),
    "Malaria_MSRA": os.path.join(folder_path, "Malaria- maternal-sexual-and-reproductive-health.csv"),
    "Screening": os.path.join(folder_path, "MSRH - screening.csv"),
    "Postpartum_Care": os.path.join(folder_path, "MSRH- Postpartum care.csv")
}

# Check if files exist before loading
for name, path in file_paths.items():
    if not os.path.exists(path):
        print(f"Error: File not found -> {path}")

# Load only existing datasets
datasets = {name: pd.read_csv(path, encoding="utf-8") for name, path in file_paths.items() if os.path.exists(path)}

# Display dataset information
for name, df in datasets.items():
    print(f"Dataset: {name}")
    print(df.info())  # Display column names and data types
    print(df.head(), "\n")  # Preview first few rows
