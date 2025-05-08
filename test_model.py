from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "./t5-medical-model/checkpoint-60"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Ensure special tokens are correctly handled
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to end-of-sequence token
tokenizer.padding_side = "left"  # Optional: Align padding to the left side

def generate_advice(input_text):
    input_text = "give medical advice: " + input_text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)

    # Enable sampling to allow temperature to affect output
    output = model.generate(
        inputs,
        max_length=256,  # Increased max length
        num_beams=4,
        do_sample=True,  # Enable sampling
        early_stopping=True,
        repetition_penalty=2.0,
        no_repeat_ngram_size=3,
        temperature=0.9
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result

user_input = input("Enter scenario description: ")
advice = generate_advice(user_input)
print("\nGenerated Medical Advice:\n", advice)
