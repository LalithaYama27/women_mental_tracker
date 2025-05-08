from llama_cpp import Llama


llm = Llama(model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf")

# Test the model
response = llm("Give me mental health tips for women.")

# Print the output
print(response)
