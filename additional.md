


As discussed above, the GRPO algorithm involves three main steps:
1. Group Sampling: Generate multiple responses for each question. Then evaluate the responses based on the reward model (reward scoring).
    -  This reward model/function can be:
        - A Simple rule-based reward func that assigns rewards based on the correctness of the response.
        - An NN-based network reward model that can be trained to assign rewards based on the correctness of the response.
2. Advantage Calculation: Calculate the advantage value for each response.
3. Policy Update: Update the policy model based on the advantage values.

## 📝 Note on Reward Model
- The reward model can be a simple rule-based model that assigns rewards based on the correctness of the response.
- Alternatively, it can be an NN-based network reward model that can be trained to assign rewards based on the correctness of the response.
-Currently TRL supports all combinations of reward models, including rule-based reward models and NN-based reward models, mixed of both, or even the scenario that we have reward model for *some* of the samples in the dataset and not for others (Multi-task reward model). This flexibility allows the user to choose the most suitable reward model for their specific task. For example, imagine a scenario where the user has a dataset of mixed promts like math reasoning, code generation, and text generation. However, the user has only a rule-based reward model for math reasoning and **NOT** for code generation. In this case, the user can use a Multi-task reward model schema supported in TRL free of stress for crash because of a missing reward model for code generation. But Note that we always need to have at least one corresponding reward model for the samples in the dataset. 

# 🐍 Example of Multi-task rule-based reward model
```python

def format_reward_func(completions, target, **kwargs):
    """
    Evaluates completions based solely on correct format:
    Format: <think>...</think><answer>...</answer>
    """
    rewards = []
    for completion in completions:
        try:
            completion = "<think>" + completion if not completion.startswith("<think>") else completion
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def simple_math_reward_func(completions, target, nums, **kwargs):
    """
    Evaluates completions based on:
    1. Correct format: <think>...</think><answer>...</answer>
    2. Uses numbers from the provided set (subset allowed)
    3. Evaluates to the target value
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # Enforce full format
            completion = "<think>" + completion if not completion.startswith("<think>") else completion
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
                continue
                
            equation = match.group(2).strip()
            used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
            
            # Check if all used numbers are in the provided set (subset allowed)
            if not all(num in numbers for num in used_numbers):
                rewards.append(0.0)
                continue
                
            # Check for allowed characters only
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue
                
            # Evaluate the equation safely
            result = eval(equation, {"__builtins__": None}, {})
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards
def code_reward_func(completions, **kwargs):
    """
    Evaluates the correctness of code snippets inside <code>...</code>.

    Args:
        completions (list[str]): Generated outputs
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    
    for completion in completions:
        try:
            match = re.search(r"<code>([\s\S]+?)<\/code>", completion)
            if match is None:
                rewards.append(0.0)
                continue

            code = match.group(1).strip()

            # Check for disallowed imports and unsafe execution patterns
            if re.search(r"\b(import|exec|eval|open|os|sys|subprocess)\b", code):
                rewards.append(0.0)
                continue

            # Validate syntax using AST (Abstract Syntax Tree)
            try:
                ast.parse(code)
            except SyntaxError:
                rewards.append(0.3)  # Partial reward for format but incorrect syntax
                continue

            rewards.append(1.0)  # Fully correct if it passes format and syntax checks

        except Exception:
            rewards.append(0.0)  # Fail-safe
        
    return rewards
```
Note: `<think>` is synthetically added to each of the prompt by us as discussed in the original deepseek papar. 

![img](./img/1.png)

We can simply test the above functions by;
```python
correct_sample_3 = """Let's use the numbers 4, 8, 2, and 1 to get 15. 
We can try: 8 * 2 - 1 = 15... </think>
<answer> 8 * 2 - 1 </answer>"""

correct_sample_4 = """ ... </think>
<answer> 8 * 2 - 1 </answer>"""

wrong_format_3 = """Using 4, 8, 2, 1, make 15: 8 * 2 - 1 = 15"""

wrong_format_4 = """<answer> 8 + 4 + 2 + 1 </answer>
<think> This should equal 15 </think>"""

wrong_result_2 = """ ... </think>
<answer> 8 + 4 - 2 </answer>"""

wrong_numbers = """ ... </think>
<answer> 10 * 2 - 5 </answer>"""

# Test setup
test_completions = [
    correct_sample_3,
    correct_sample_4,
    wrong_format_3,
    wrong_format_4,
    wrong_result_2,
    wrong_numbers
]
test_target = ["15"] * 6
test_nums = [[4, 8, 2, 1]] * 6

# Run tests
format_rewards = format_reward_func(test_completions, test_target)
print("Format rewards:", format_rewards)
assert format_rewards == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0], "Format reward function is not working"

math_rewards = simple_math_reward_func(test_completions, test_target, nums=test_nums)
print("Math rewards:", math_rewards)
assert math_rewards == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0], "Math reward function is not working"
```

This looks good, now we can move to the next step of defining training parameters of the GRPO algorithm.

```python
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig

# our model we are going to use as policy 
model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
    use_peft=True,
    load_in_4bit=True,
)

# Hyperparameters
training_args = GRPOConfig(
    output_dir="qwen-r1-tiny",
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    logging_steps=10,
    max_steps=100,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    # GRPO specific parameters
    max_prompt_length=256,
    max_completion_length=1024, # max length of the generated output 
    num_generations=2, # the minimum is two, CANNOT be less than 2 (non-sense for averaging)
    beta=0.001, # was 0.04 in the original paper
    
)
trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[simple_math_reward_func, math_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=get_peft_config(model_config),
)
trainer.train()
```

As you see in the above training argument the `reward_funcs=[simple_math_reward_func, math_reward_func]` though we also have a `code_reward_func` but we are not using it in this training cause for instance the dataset we picked for training does not have code samples prompts. But this will not cause any crash or error, the code will run smoothly. 😎

# 🔥 Example of Multiple Mixed reward functions

```python
# GRPO trainer can handle a mix of reward functions and reward models in the same training run
from datasets import load_dataset

dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

def reward_func(completions, **kwargs):
    """Reward function that rewards completions longer than the batch average."""
    avg_length = sum(len(c) for c in completions) / len(completions) if completions else 1
    return [float(len(c) / avg_length) for c in completions]

training_args = GRPOConfig(
    output_dir=tmp_dir,
    learning_rate=0.1,  # increase the learning rate to speed up the test
    per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
    num_generations=3,  # reduce the number of generations to reduce memory usage
    max_completion_length=32,  # reduce the completion length to reduce memory usage
    report_to="none",
    )
trainer = GRPOTrainer(
    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    reward_funcs=[reward_func, "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"],
    args=training_args,
    train_dataset=dataset,
    )

trainer.train()
```

As you see here we have a mix of reward functions and reward models, the `reward_func` is a simple rule-based reward function that rewards longer completions, while the second reward model is a pre-trained model that assigns rewards based on the correctness of the response. This flexibility allows the user to choose the most suitable reward model for their specific task.
