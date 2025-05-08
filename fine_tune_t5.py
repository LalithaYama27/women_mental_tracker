import os
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
from transformers import EarlyStoppingCallback
from evaluate import load as load_metric

# Load dataset
dataset = load_dataset("csv", data_files={"train": r"C:\Users\welcome\Downloads\women_mental_tracker\train_data.csv",
                                          "validation": r"C:\Users\welcome\Downloads\women_mental_tracker\val_data.csv"})

# Load tokenizer and model
model_checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

# Preprocessing
max_input_length = 256
max_target_length = 128

def preprocess_function(examples):
    inputs = examples["input_text"]
    targets = examples["target_text"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Evaluation metric
bleu = load_metric("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = [[token if token != -100 else tokenizer.pad_token_id for token in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Simple postprocessing
    decoded_preds = [pred.strip().split() for pred in decoded_preds]
    decoded_labels = [[label.strip().split()] for label in decoded_labels]

    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["bleu"]}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train model
trainer.train()

# Save model
trainer.save_model("./t5_finetuned_final")

# Evaluate
results = trainer.evaluate(tokenized_datasets["test"])
print("Evaluation Results:", results)

# Generate predictions
test_loader = tokenized_datasets["test"]
predictions = trainer.predict(test_loader)
decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode([[token if token != -100 else tokenizer.pad_token_id for token in label] for label in predictions.label_ids], skip_special_tokens=True)

# Print a few samples
for i in range(5):
    print(f"\nInput: {test_loader[i]['input_text']}")
    print(f"Target: {test_loader[i]['target_text']}")
    print(f"Prediction: {decoded_preds[i]}")
