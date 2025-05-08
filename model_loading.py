from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load your custom dataset (assuming it's a CSV or JSON file with columns 'input_text' and 'target_text')
dataset = load_dataset("csv", data_files="train_data.csv")  # Update this with your actual dataset file

# Convert the dataset into a pandas DataFrame (this helps with train_test_split)
dataset_df = dataset['train'].to_pandas()

# Now split the data into train and validation sets
train_dataset, val_dataset = train_test_split(dataset_df, test_size=0.2)

# Print the shape of the split datasets
print(f"Train Dataset: {train_dataset.shape}")
print(f"Validation Dataset: {val_dataset.shape}")

# Load the pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')  # You can change this to 't5-base' or 't5-large' for larger models
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Example of how you can process the data and fine-tune the model
# You can replace this with your own processing logic
def process_data(dataframe):
    # Tokenize the inputs and targets
    inputs = tokenizer(list(dataframe['input_text']), padding=True, truncation=True, return_tensors="pt")
    targets = tokenizer(list(dataframe['target_text']), padding=True, truncation=True, return_tensors="pt")

    # Return the tokenized inputs and targets
    return inputs, targets

train_inputs, train_targets = process_data(train_dataset)
val_inputs, val_targets = process_data(val_dataset)

# Training loop (example, replace with your actual training loop)
def train(model, train_inputs, train_targets):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(3):  # Example, you can increase the epochs
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=train_inputs['input_ids'], labels=train_targets['input_ids'])
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1} - Loss: {loss.item()}")

# Fine-tune the model (using the prepared data)
train(model, train_inputs, train_targets)

# Save the fine-tuned model
model.save_pretrained("fine_tuned_t5")

# Save the tokenizer as well
tokenizer.save_pretrained("fine_tuned_t5")
