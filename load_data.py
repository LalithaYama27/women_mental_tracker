import pandas as pd
import json5

# File Paths
file_paths = {
    "General": r"C:\Users\welcome\Downloads\women_mental_tracker\cleaned_General - MSRH.csv",
    "Local Stories": r"C:\Users\welcome\Downloads\women_mental_tracker\cleaned_Local Stories on Maternal Sexual and Reproductive Health in Africa.csv",
    "Malaria": r"C:\Users\welcome\Downloads\women_mental_tracker\cleaned_Malaria- maternal-sexual-and-reproductive-health.csv",
    "Screening": r"C:\Users\welcome\Downloads\women_mental_tracker\cleaned_MSRH - screening.csv",
    "Postpartum Care": r"C:\Users\welcome\Downloads\women_mental_tracker\cleaned_MSRH- Postpartum care.csv"
}

# Load datasets
datasets = {}
for name, path in file_paths.items():
    try:
        datasets[name] = pd.read_csv(path, encoding='utf-8')
        print(f"✔ Successfully loaded {name} dataset.")
    except Exception as e:
        print(f" Error loading {name}: {e}")

# Load prompts.json
prompts_path = r"C:\Users\welcome\Downloads\women_mental_tracker\prompts.json"

try:
    with open(prompts_path, "r", encoding="utf-8") as file:
        prompts = json5.load(file)
        print("✔ Successfully loaded prompts.json")
except Exception as e:
    print(f" Error loading prompts.json: {e}")

# Print sample data
for name, df in datasets.items():
    print(f"\n Sample from {name} dataset:")
    print(df.head(), "\n")

print("\n Sample prompts from prompts.json:")
print(json5.dumps(prompts, indent=4))

