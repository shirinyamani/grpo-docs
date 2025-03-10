# Core intuition of GRPO

##### **Goal**
**By comparing the completion generated within groups** by the policy model, rather than training the value model (Critic), leading to significant reduction of computational cost!

##### **Application**: 
mostly in verifiable domains like Math reasoning or/and code generation that requires clear reward rules cause this is a rule-based reward scenario

# Steps of GRPO
### Step 1) **Group Sampling**:
#### **Action:** 
For each question $q$, the model will generate $G$ outputs (group size) from the old policy model: {${o_1, o_2, o_3, \dots, o_G}$} $\pi_{\theta_{\text{old}}}$ , $G=8$ where each $o_i$ represents one completion from the model.
#### **Example**:
- **Question** 
	- $q$ : $\text{Calculate}\space2 + 2 \times 6$
- **Output**: we will have $8$ responses; $(G = 8)$	$${o_1:14(correct), o_2:10 (wrong), o_3:16 (wrong), ... o_G:14(correct)}$$
### Step 2) **Advantage calculation**:
#### **Reward Distribution:**
Assign a RM score to each of the generated responses based on the correctness $r_i$ *(e.g. 1 for correct response, 0 for wrong response)* then for each of the $r_i$ calculate the following Advantage value 
#### **Advantage value formula**:
**Formula**:$$A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \ldots, r_G\})}{\text{std}(\{r_1, r_2, \ldots, r_G\})}$$
#### **Example**:
for the same example above, imagine we have 8 responses, 4 of which is correct and the rest wrong, therefore;
- Group Average: $mean(r_i) = 0.5$
- Std: $std(r_i) = 0.53$
- Advantage Value:
	- Correct response: $A_i = \frac{1 - 0.5}{0.53}= 0.94$
	- Wrong response: $A_i = \frac{0 - 0.5}{0.53}= -0.94$
##### **Meaning**:  
- This standardization (i.e. $A_i$ weighting) allows the model to assess each response's relative performance, guiding the optimization process to favour responses that are better than average (high reward) and discourage those that are worse.  For instance if $A_i > 0$, then the $o_i$ is better response than the average level within it's group; and if if $A_i < 0$, then the $o_i$ then the quality of the response is less than the average (i.e. poor quality/performance). 
- For the example above, if $A_i = 0.94 \text{(correct output)}$ then during optimization steps its generation probability will be increased. 
### Step 3) **Policy Update:** 
#### **Target Function**:
$$J_{GRPO}(\theta) = \left[\frac{1}{G} \sum_{i=1}^{G} \min \left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i \text{clip}\left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1 - \epsilon, 1 + \epsilon \right) A_i \right)\right]- \beta D_{KL}(\pi_{\theta} || \pi_{ref})$$

#### **Key components of the Target function**:
##### **1. Probability ratio:**   $\left(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}\right)$ 
Intuitively, the formula compares how much the new model's response probability differs from the old model's response probability while incorporating a preference for responses that improve the expected outcome.
###### **Meaning**:
- If $\text{ratio} > 1$, the new model assigns a higher probability to response $o_i$​ than the old model.
- If $\text{ratio} < 1$, the new model assigns a lower probability to $o_i$​ 
##### **2. Clip function:** $\text{clip}\left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1 - \epsilon, 1 + \epsilon \right)$ 
Limit the ratio discussed above to be within $[1 - \epsilon, 1 + \epsilon]$ to avoid/control drastic changes or crazy updates and stepping too far off from the old policy. In other words, it limit how much the probability ratio can increase to help maintaining stability by avoiding updates that push the new model too far from the old one.
##### **Example** $\space \text{suppose}(\epsilon = 0.2)$
- **Case 1**: if the new policy has a probability of 0.9 for a specific response and the old policy has a probabiliy of 0.5, it means this response is getting reiforeced by the new policy to have higher probability, but within a controlled limit which is the clipping to tight up its hands to not get drastic 
	- $\text{Ratio}: \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} = \frac{0.9}{0.5} = 1.8  → \text{Clip}\space1.2$ (upper bound limit 1.2) 
- **Case 2**: If the new policy is not in favour of a response (lower probability e.g. 0.2), meaning if the response is not beneficial the increase might be incorrect, and the model would be penalized.
	- $\text{Ratio}: \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} = \frac{0.2}{0.5} = 0.4  →\text{Clip}\space0.8$ (lower bound limit 0.8)
###### **Meaning**:
- The formula encourages the new model to favour responses that the old model underweighted **if they improve the outcome**.
- If the old model already favoured a response with a high probability, the new model can still reinforce it **but only within a controlled limit $[1 - \epsilon, 1 + \epsilon]$, $\text{(e.g., }\epsilon = 0.2, \space \text{so} \space [0.8-1.2])$**.
- If the old model overestimated a response that performs poorly, the new model is **discouraged** from maintaining that high probability.
- Therefore, intuitively, By incorporating the probability ratio, the objective function ensures that updates to the policy are proportional to the advantage $A_i$ while being moderated to prevent drastic changes. T

#### **3. KL Divergence:**  $\beta D_{KL}(\pi_{\theta} || \pi_{ref})$
KL Divergence is used to prevent over-optimization of the reward model, which in this context is refers to when the model output non-sensical text or in our math reasoning example, the model will generate extremely incorrect answers!
##### **Example**
Suppose the reward model has a flaw—it **wrongly assigns higher rewards to incorrect outputs** due to spurious correlations in the training data. So,  $2 + 2 \times 6 = 20$ then later the Ratio $R(o_6​=20)=0.95$ *(wrong but rewarded highly)*, without KL Divergence, during optimization, the model will learn to favours responses that are higher numbers, assuming they indicate *"more confident"* reasoning, i.e. the model starts shifting its policy towards these outputs. And future iterations reinforce these incorrect answers. So, say  $2 + 2 \times 6 = 42 \space \text{(a random common number in datasets)}$. This response doesn't even resemble arithmetic errors anymore. Instead, the model has learned to exploit whatever patterns maximize the reward signal, regardless of correctness. 
##### **Meaning**
- A KL divergence penalty keeps the model’s outputs close to its original distribution, preventing extreme shifts.
- Even if incorrect answers receive high rewards, the model cannot deviate too much from what it originally considered reasonable.
- Instead of drifting towards completely irrational outputs, the model would refine its understanding while still allowing some exploration
-  The 
##### **Math Definition**
Recall that KL distance is defined as follows:
$$D_{KL}(P || Q) = \sum_{x \in X} P(x) \log \frac{P(x)}{Q(x)}$$
In RLHF, the two distributions of interest are often the distribution of the new model version, say $P(x)$, and a distribution of the reference policy, say $Q(x)$.
##### **Term** $\beta \space$ in $\beta D_{KL}(\pi_{\theta} || \pi_{ref})$
-  **Higher $\beta$ (Stronger KL Penalty)**
    - More constraint on policy updates. The model remains close to its reference distribution.
    - Can slow down adaptation: The model may struggle to explore better responses.
- **Lower $\beta$ (Weaker KL Penalty)**
    - More freedom to update policy: The model can deviate more from the reference.
    - Faster adaptation but risk of instability: The model might learn reward-hacking behaviors.
	- Over-optimization risk: If the reward model is flawed, the policy might generate nonsensical outputs.
- **Original** [DeepSeekMath](https://arxiv.org/abs/2402.03300) paper set this $\beta= 0.04$

# Complete Simple Math Example
#### **Question** 
$$\text{Q: Calculate}\space2 + 2 \times 6$$

#### **Step 1) Group sampling**
Generate $(G = 8)$ responses, $4$ of which are correct answer ($14, \text{reward=} 1$) and $4$ incorrect $\text{(reward= 0)}$, Therefore:

$${o_1:14(correct), o_2:10 (wrong), o_3:16 (wrong), ... o_G:14(correct)}$$
#### **Step 2) Advantage Calculation**
Group Average: 
$$mean(r_i) = 0.5$$
- Std: $$std(r_i) = 0.53$$
- Advantage Value:
	- Correct response: $A_i = \frac{1 - 0.5}{0.53}= 0.94$
	- Wrong response: $A_i = \frac{0 - 0.5}{0.53}= -0.94$
#### **Step 3) Policy Update**
- Assuming the probability of old policy ($\pi_{\theta_{old}}$) for a correct output $o_1$ is $0.5$ and the new policy increases it to $0.7$ then:
$$\text{Ratio}: \frac{0.7}{0.5} = 1.4  →\text{after Clip}\space1.2 \space (\epsilon = 0.2)$$
- Then when the target function is re-weighted, the model tends to reinforce the generation of correct output, and the $\text{KL Divergence}$  limits the deviation from the reference policy. 


