from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load the same fine-tuned model multiple times (simulate different models)
model_1 = T5ForConditionalGeneration.from_pretrained("./model_checkpoint_1").to("cuda" if torch.cuda.is_available() else "cpu")
model_2 = T5ForConditionalGeneration.from_pretrained("./model_checkpoint_2").to("cuda" if torch.cuda.is_available() else "cpu")
