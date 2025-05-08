import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from bert_score import score as bert_score
from tqdm import tqdm


def evaluate_model(model_path, val_data_path):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load validation data
    val_df = pd.read_csv(val_data_path)
    if 'input_text' not in val_df.columns or 'target_text' not in val_df.columns:
        raise ValueError("CSV must have 'input_text' and 'target_text' columns.")

    inputs = val_df['input_text'].tolist()
    targets = val_df['target_text'].tolist()
    predictions = []

    print("\nGenerating predictions...\n")

    for i, text in enumerate(tqdm(inputs, desc="Evaluating")):
        # Tokenize input
        encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate output
        with torch.no_grad():
            output = model.generate(**encoded_input, max_length=100)

        # Decode and normalize
        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
        predictions.append(pred)

        print(f"\nInput    : {text}")
        print(f"Predicted: {pred}")
        print(f"Target   : {targets[i].strip().lower()}")

    # Normalize references and candidates
    y_true = [t.strip().lower() for t in targets]
    y_pred = [p.strip().lower() for p in predictions]

    # BLEU Score
    bleu_score_val = corpus_bleu([[ref.split()] for ref in y_true], [pred.split() for pred in y_pred])
    print(f"\nBLEU Score: {bleu_score_val:.4f}")

    # BERTScore
    print("\nCalculating BERTScore...")
    P, R, F1 = bert_score(cands=y_pred, refs=y_true, lang='en', rescale_with_baseline=True)
    print(f"BERTScore Precision: {P.mean():.4f}")
    print(f"BERTScore Recall   : {R.mean():.4f}")
    print(f"BERTScore F1       : {F1.mean():.4f}")

    # Optional: use exact string match metrics for short text only
    if all(len(t.split()) < 10 for t in y_true):
        print("\n⚠ Using classification metrics for short labels (not recommended for long text generation).")
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print("\nEvaluation Metrics:")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
    else:
        print("\nSkipped classification metrics: predictions are likely long text, not discrete class labels.")


# === Replace with your actual model and validation CSV paths ===
model_path = r"C:\Users\welcome\Downloads\women_mental_tracker\flan-t5-finetuned"
val_data_path = r"C:\Users\welcome\Downloads\women_mental_tracker\val_data.csv"

evaluate_model(model_path, val_data_path)
