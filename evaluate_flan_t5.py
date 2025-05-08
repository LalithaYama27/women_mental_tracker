import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import evaluate

# Load FLAN-T5 model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()

# Load validation data
val_data = pd.read_csv("val_data.csv")

# Prepare inputs
inputs = ["give medical advice: " + text for text in val_data['input_text']]
input_encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=256)

# Generate predictions
with torch.no_grad():
    outputs = model.generate(**input_encodings, max_length=128)

predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
references = val_data['target_text'].tolist()

# Compute BLEU
bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])

# Compute ROUGE
rouge = evaluate.load("rouge")
rouge_score = rouge.compute(predictions=predictions, references=references)

# Output metrics
print("\nEvaluation Metrics:")
print(f"BLEU Score: {bleu_score['bleu']:.4f}")
print(f"ROUGE-1: {rouge_score['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_score['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
