from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch

# Load dataset
data_files = {
    "train": "train_data.csv",
    "validation": "val_data.csv"
}
dataset = load_dataset("csv", data_files=data_files)

# Check column names to make sure "Scenario" and "Medical solution" exist
print(dataset['train'].column_names)

# Load pre-trained tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust num_labels based on your task

# Preprocessing function
max_length = 256  # Adjust max length based on your dataset

def preprocess(example):
    # Make sure "Scenario" and "Medical solution" are the correct column names
    inputs = example["Scenario"]  # Change this if the column name is different
    targets = example["Medical solution"]  # Change this if the column name is different
    
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=max_length)

    # Labels are the target sequence (Medical solution)
    labels = tokenizer(targets, truncation=True, padding="max_length", max_length=max_length)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Apply preprocessing to the dataset
tokenized_dataset = dataset.map(preprocess, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="./bert-medical-model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=20,
    save_strategy="epoch",
)

# Define metrics (simple accuracy)
def compute_metrics(p):
    logits, labels = p
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == labels).sum().item() / len(labels)
    return {"accuracy": accuracy}

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./bert-medical-model')
tokenizer.save_pretrained('./bert-medical-model')

print("Model and tokenizer saved successfully!")
