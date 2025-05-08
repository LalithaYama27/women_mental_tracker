from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned model from the folder where you saved it
model = T5ForConditionalGeneration.from_pretrained('./t5-medical-model')

# Load the tokenizer from the same folder
tokenizer = T5Tokenizer.from_pretrained('./t5-medical-model')

print("Model and tokenizer loaded successfully!")
