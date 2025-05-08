from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import evaluate
import numpy as np

# Load the dataset we saved earlier
data_files = {
    "train": "train_data.csv",
    "validation": "val_data.csv"
}
dataset = load_dataset("csv", data_files=data_files)

# Load pre-trained tokenizer and model
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocessing
max_input_length = 256
max_target_length = 128

def preprocess(example):
    inputs = ["give medical advice: " + x for x in example["input_text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

    labels = tokenizer(example["target_text"], max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="./t5-medical-model",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=20,
    save_strategy="epoch",
    
)

# Metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace for comparison
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Simple accuracy calculation
    correct = sum([pred == label for pred, label in zip(decoded_preds, decoded_labels)])
    total = len(decoded_preds)

    return {"accuracy": correct / total}

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./t5-medical-model')
tokenizer.save_pretrained('./t5-medical-model')

print("Model and tokenizer saved successfully!")
