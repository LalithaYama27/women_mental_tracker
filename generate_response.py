from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer from the correct path
model = T5ForConditionalGeneration.from_pretrained('./saved_model_pretrained')
tokenizer = T5Tokenizer.from_pretrained('t5-small')  # or the base model you used

# Input prompt
input_text = "summarize: Women face unique challenges in accessing reproductive healthcare services in rural areas."

# Tokenize and generate
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
output = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)

# Decode and print result
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Response:", response)
