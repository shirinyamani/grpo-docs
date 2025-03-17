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
prompt = "Solve y = 2x + 1 for x = 2, y = "  # Correct answer: 5
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(device)  # Shape: (1, prompt_len)
attention_mask = inputs["attention_mask"].to(device)

# Step 1: Generate 8 responses (B = 2 groups, G = 4 responses per group)
batch_size, num_generations = 2, 4
outputs = model.generate(
    input_ids=input_ids,  # Shape: (1, prompt_len)
    attention_mask=attention_mask,
    max_new_tokens=1,  # L = 1 (single token per response)
    num_return_sequences=batch_size * num_generations,  # 8 responses total
    do_sample=True,
    top_k=10,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True
)

# Extract generated tokens and log probs
gen_ids = outputs.sequences[:, -1:]  # Shape: (8, 1)
logits = outputs.scores[0]  # Shape: (8, vocab_size) for L=1
per_token_logps = F.log_softmax(logits, dim=-1)  # Shape: (8, vocab_size)
per_token_logps = per_token_logps.gather(1, gen_ids)  # Shape: (8, 1), on cuda:0

# Decode and compute rewards
responses = []
rewards = []
print("\nStep 1: Group Sampling (B = 2, G = 4)")
for i, gen_id in enumerate(gen_ids[:, 0]):
    group_idx = i // num_generations + 1
    resp_idx = i % num_generations + 1
    raw_response = tokenizer.decode(gen_id, skip_special_tokens=True).strip()
    try:
        response_val = float(raw_response)
    except ValueError:
        response_val = 0.0
    responses.append(response_val)
    reward = 1.0 if abs(response_val - 5.0) < 0.5 else 0.0
    rewards.append(reward)
    print(f"Group {group_idx}, o{resp_idx}: {response_val} ({'correct' if reward == 1.0 else 'wrong'})")
responses = torch.tensor(responses, device=device)  # Shape: (8,), on cuda:0
rewards = torch.tensor(rewards, device=device)      # Shape: (8,), on cuda:0

# Step 2 & 3: GRPO Processing
def grpo_process(responses, rewards, per_token_logps, batch_size=2, num_generations=4):
    B, G = batch_size, num_generations
    
    # Step 2: Advantage Calculation
    print("\nStep 2: Advantage Calculation")
    rewards_grouped = rewards.view(B, G)  # Shape: (2, 4)
    mean_grouped_rewards = rewards_grouped.mean(dim=1)  # Shape: (2,)
    std_grouped_rewards = rewards_grouped.std(dim=1)    # Shape: (2,)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(G, dim=0)  # Shape: (8,)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(G, dim=0)    # Shape: (8,)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)  # Shape: (8,)
    advantages = advantages.unsqueeze(1)  # Shape: (8, 1), on cuda:0
    print(f"Grouped Rewards:\n{rewards_grouped}")
    print(f"Mean per group: {mean_grouped_rewards[:G]} (Group 1), {mean_grouped_rewards[G:]} (Group 2)")
    print(f"Std per group: {std_grouped_rewards[:G]} (Group 1), {std_grouped_rewards[G:]} (Group 2)")
    print(f"Advantages: {advantages.T}")
    
    # Step 3: Policy Update with PPO
    print("\nStep 3: Policy Update")
    print(f"Old Log Probs: {per_token_logps.T}")
    # Simulate new policy
    delta = 0.5
    new_per_token_logps = per_token_logps.clone()
    for i, r in enumerate(rewards):
        if r == 1.0:
            new_per_token_logps[i] = min(new_per_token_logps[i] + delta, 0.0)
        else:
            new_per_token_logps[i] -= delta
    print(f"New Log Probs: {new_per_token_logps.T}")
    
    # PPO objective
    ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (8, 1), on cuda:0
    eps = 0.2
    pg_losses1 = -advantages * ratio  # Shape: (8, 1)
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (8, 1)
    pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (8, 1)
    
    # KL penalty (placeholder)
    per_token_kl = torch.abs(new_per_token_logps - per_token_logps)  # Shape: (8, 1)
    beta = 0.01
    per_token_loss = pg_loss_max + beta * per_token_kl  # Shape: (8, 1)
    print(f"Ratio: {ratio.T}")
    print(f"Loss per token: {per_token_loss.T}")
    
    return rewards, per_token_logps, new_per_token_logps, per_token_loss

# Execute
rewards, per_token_logps, new_per_token_logps, per_token_loss = grpo_process(responses, rewards, per_token_logps, batch_size, num_generations)

# Final Output
print(f"\nFinal Results:")
print(f"Responses: {responses.tolist()}")
print(f"Rewards: {rewards.tolist()}")
print(f"Old Log Probs: {per_token_logps.T}")
print(f"New Log Probs: {new_per_token_logps.T}")
print(f"Mean Loss: {per_token_loss.mean().item():.3f}")