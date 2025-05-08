import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import evaluate

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('./t5-medical-model')
model = T5ForConditionalGeneration.from_pretrained('./t5-medical-model')
model.eval()

# Load validation data
val_data = pd.read_csv("val_data.csv")

# Preprocess input texts (add your instruction prefix)
inputs = ["give medical advice: " + text for text in val_data['input_text']]

# Tokenize the inputs
input_encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=256)

# Generate predictions using beam search instead of greedy decoding for better results
with torch.no_grad():
    outputs = model.generate(
        **input_encodings, 
        max_length=128,  # Adjust max_length as per your requirement
        num_beams=5,  # Use beam search
        early_stopping=True,  # Stop early when sufficient candidates are found
        no_repeat_ngram_size=2  # Avoid repeating n-grams
    )

# Decode predictions
predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Get reference text from validation data
references = val_data['target_text'].tolist()

# Compute BLEU Score
bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])

# Compute ROUGE Scores
rouge = evaluate.load("rouge")
rouge_score = rouge.compute(predictions=predictions, references=references)

# Output results
print("\nEvaluation Metrics:")
print(f"BLEU Score: {bleu_score['bleu']:.4f}")
print(f"ROUGE-1: {rouge_score['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_score['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
