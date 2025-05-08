from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Trainer, TrainingArguments
import numpy as np
import evaluate

# Load dataset
data_files = {
    "train": "train_data.csv",
    "validation": "val_data.csv"
}
dataset = load_dataset("csv", data_files=data_files)

# Load tokenizer and model (FLAN-T5)
model_name = "google/flan-t5-small"
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

# Training arguments
training_args = TrainingArguments(
    output_dir="./flan-t5-finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=3e-4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

# Accuracy metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions[0], axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    correct = sum([p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)])
    return {"accuracy": correct / len(decoded_preds)}

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model)

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

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained("./flan-t5-finetuned")
tokenizer.save_pretrained("./flan-t5-finetuned")

print("Fine-tuned FLAN-T5 model saved.")
