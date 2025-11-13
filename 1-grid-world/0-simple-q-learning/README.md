# Simple Q-Learning

This is a basic implementation of the **Q-Learning algorithm**, one of the most fundamental reinforcement learning algorithms. It demonstrates how an agent can learn to navigate from any state to a goal state using only rewards.

## 🎯 What is Q-Learning?

Q-Learning is a **model-free** reinforcement learning algorithm that learns the value of taking an action in a particular state. The "Q" stands for "quality" - it measures the quality of a state-action pair.

### Key Concepts

- **Q-Table**: A lookup table where rows represent states and columns represent actions. Each cell contains the Q-value for that state-action pair.
- **Reward Matrix (R)**: Defines immediate rewards for state transitions
- **Bellman Equation**: The core update rule for Q-Learning

### The Q-Learning Update Rule

```
Q(s,a) ← Q(s,a) + α * [R(s,a) + γ * max(Q(s',a')) - Q(s,a)]
```

Where:
- `Q(s,a)` = Q-value for state `s` and action `a`
- `α` (alpha) = Learning rate (0-1)
- `γ` (gamma) = Discount factor (0-1)
- `R(s,a)` = Immediate reward
- `max(Q(s',a'))` = Maximum Q-value for next state `s'`

## 🏗️ Problem Description

In this example, we have:
- **6 states** (0, 1, 2, 3, 4, 5)
- **Goal**: Reach state 5 from any starting state
- **Reward Matrix**: Defines valid transitions and rewards

### Reward Matrix

```
     0   1   2   3   4   5
0 [ -1, -1, -1, -1,  0, -1 ]
1 [ -1, -1, -1,  0, -1, 100]
2 [ -1, -1, -1,  0, -1, -1 ]
3 [ -1,  0,  0, -1,  0, -1 ]
4 [  0, -1, -1,  0, -1, 100]
5 [ -1,  0, -1, -1,  0, 100]
```

- `-1`: Action not allowed (no transition)
- `0`: Valid transition with no immediate reward
- `100`: Reaching the goal state

## 🚀 How to Run

```bash
python simple_q_learning.py
```

### Expected Output

1. **Training Progress**: Shows episode completion
2. **Learned Q-Table**: Normalized Q-values (0-100 scale)
3. **Optimal Paths**: Best path from each state to goal
4. **Visualization**: Heatmap of the Q-table saved as `q_table_heatmap.png`

## 📊 Example Output

```
Simple Q-Learning Example
==========================================================

Reward Matrix:
[[ -1  -1  -1  -1   0  -1]
 [ -1  -1  -1   0  -1 100]
 [ -1  -1  -1   0  -1  -1]
 [ -1   0   0  -1   0  -1]
 [  0  -1  -1   0  -1 100]
 [ -1   0  -1  -1   0 100]]

Goal: Learn to reach state 5 from any starting state
----------------------------------------------------------
Episode 200/1000 completed
Episode 400/1000 completed
Episode 600/1000 completed
Episode 800/1000 completed
Episode 1000/1000 completed
Training completed!

Learned Q-Table (normalized to 0-100):
[[  0   0   0   0  64   0]
 [  0   0   0  80   0 100]
 [  0   0   0  64   0   0]
 [  0  80  64   0  80   0]
 [ 64   0   0  80   0 100]
 [  0  80   0   0  80 100]]

Testing Optimal Paths:
State 0 -> 0 -> 4 -> 5
State 1 -> 1 -> 5
State 2 -> 2 -> 3 -> 4 -> 5
State 3 -> 3 -> 4 -> 5
State 4 -> 4 -> 5
```

## 📈 Understanding the Results

After training:
- The agent learns that state 5 is the goal (Q-values of 100)
- It discovers optimal paths from any state to the goal
- The Q-table shows the "quality" of each action in each state
- Higher Q-values indicate better actions

## 🧮 Parameters

You can adjust these hyperparameters in the code:

- **gamma (γ)**: `0.8` - Discount factor
  - Higher values (→1): Agent values future rewards more
  - Lower values (→0): Agent prefers immediate rewards

- **alpha (α)**: `0.9` - Learning rate
  - Higher values: Fast learning but may be unstable
  - Lower values: Slower but more stable learning

- **num_episodes**: `1000` - Number of training episodes
  - More episodes generally lead to better learning

## 🔍 Why Start Here?

This example is perfect for learning because:

1. **No Neural Networks**: Pure tabular Q-Learning
2. **Simple Environment**: Easy to understand and visualize
3. **Fast Training**: Converges in seconds
4. **Foundation**: Understanding this helps with advanced RL algorithms

## 📚 Next Steps

After understanding this example, explore:

1. **Policy Iteration** - Value-based method with explicit policy
2. **Value Iteration** - Similar to Q-Learning but different update
3. **SARSA** - On-policy alternative to Q-Learning
4. **Deep Q-Network (DQN)** - Q-Learning with neural networks

## 🔗 References

- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [Original Q-Learning Paper (Watkins, 1989)](https://link.springer.com/article/10.1007/BF00992698)

## 💡 Key Takeaways

- Q-Learning learns optimal policies without knowing the environment model
- It uses trial-and-error to discover the best actions
- The Q-table stores learned knowledge about state-action values
- Convergence to optimal policy is guaranteed with proper parameters
