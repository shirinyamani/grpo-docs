import re


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

print("All tests passed!")