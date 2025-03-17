import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "Qwen/Qwen2-Math-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input prompt
prompt = "Just Solve y = 2x + 1 for x = 2, what is y? "  # Correct answer: 5
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Generate 4 outputs
outputs = model.generate(
    input_ids, 
    attention_mask=attention_mask,  # Include attention mask
    max_length=50, 
    num_return_sequences=2,  # Generates 4 different completions
    do_sample=True  # Enables sampling for diverse outputs
)

# Decode and print outputs
for i, output in enumerate(outputs):
    print(f"Generation {i+1}: {tokenizer.decode(output, skip_special_tokens=True)}")

