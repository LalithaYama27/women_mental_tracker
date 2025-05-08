from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the dataset
dataset = load_dataset("csv", data_files={
    "train": "C:/Users/welcome/Downloads/women_mental_tracker/train_data.csv",  # Replace with your file path
    "validation": "C:/Users/welcome/Downloads/women_mental_tracker/val_data.csv"  # Replace with your file path
})

# Print column names to ensure you're using the correct ones
print(dataset["train"].column_names)

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Preprocess the data
def preprocess_function(examples):
    # Use 'input_text' and 'target_text' as column names based on your dataset
    inputs = examples['input_text']
    targets = examples['target_text']
    
    model_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=512)  # Adjust max_length as needed
    labels = tokenizer(targets, padding=True, truncation=True, max_length=512)  # Adjust max_length as needed
    model_inputs['labels'] = labels['input_ids']  # Ensure labels are set correctly
    return model_inputs

# Apply preprocessing
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # or "steps", depending on your preference
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_steps=500,  # Save the model after every 500 steps
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the model and tokenizer after training
model.save_pretrained('C:/Users/welcome/Downloads/women_mental_tracker/saved_model_pretrained')  # Change this to the desired save path
tokenizer.save_pretrained('C:/Users/welcome/Downloads/women_mental_tracker/tokenizer_saved_model')  # Change this to the desired save path

# Ensure everything loads correctly
print(model)
print(tokenizer)

# Save model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# If you're using a separate model checkpoint, you can define its path too
# Example for a model saving path
trainer.save_model("./fine_tuned_model")  # This saves the model during training