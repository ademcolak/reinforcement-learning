#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Q-Learning Example
==========================
This is a basic implementation of Q-Learning algorithm using a reward matrix.
The agent learns to navigate from any state to the goal state (state 5).

Q-Learning is a model-free reinforcement learning algorithm that learns
the value of an action in a particular state. It does not require a model
of the environment and can handle problems with stochastic transitions.

Key concepts:
- Q-table: A table that stores Q-values for each state-action pair
- Reward matrix: Defines immediate rewards for state transitions
- Gamma (γ): Discount factor for future rewards
- Alpha (α): Learning rate

Author: RLCode Team
Updated: 2025 (Python 3.8+ compatible)
"""

import numpy as np
import random
import matplotlib.pyplot as plt


class SimpleQLearning:
    """
    Simple Q-Learning agent that learns to reach a goal state.

    The environment is represented by a reward matrix R where:
    - R[s, a] = reward for taking action a in state s
    - -1 means the action is not allowed
    - 0 means valid transition
    - 100 means reaching the goal
    """

    def __init__(self, reward_matrix, gamma=0.8, alpha=0.9):
        """
        Initialize the Q-Learning agent.

        Args:
            reward_matrix: NumPy array representing rewards for state-action pairs
            gamma: Discount factor (0-1), determines importance of future rewards
            alpha: Learning rate (0-1), determines how much new info overrides old
        """
        self.R = reward_matrix
        self.num_states = self.R.shape[0]
        self.Q = np.zeros_like(self.R, dtype=float)
        self.gamma = gamma
        self.alpha = alpha

    def available_actions(self, state):
        """
        Get list of valid actions from a given state.

        Args:
            state: Current state

        Returns:
            Array of valid action indices (where reward >= 0)
        """
        return np.where(self.R[state] >= 0)[0]

    def sample_next_action(self, available_actions):
        """
        Randomly sample an action from available actions.

        Args:
            available_actions: Array of valid action indices

        Returns:
            Randomly selected action
        """
        return int(np.random.choice(available_actions, size=1))

    def train(self, num_episodes=1000, verbose=True):
        """
        Train the Q-Learning agent.

        Args:
            num_episodes: Number of training episodes
            verbose: Whether to print progress
        """
        for episode in range(num_episodes):
            # Start from a random state
            state = random.randint(0, self.num_states - 1)
            available_act = self.available_actions(state)
            action = self.sample_next_action(available_act)

            # Episode loop
            while True:
                # Take action and observe next state
                next_state = action

                # Get best Q-value for next state
                best_next_actions = self.available_actions(next_state)
                if best_next_actions.size > 0:
                    max_Q_next = self.Q[next_state, best_next_actions].max()
                else:
                    max_Q_next = 0

                # Q-Learning update rule:
                # Q(s,a) = Q(s,a) + α * (R(s,a) + γ * max(Q(s',a')) - Q(s,a))
                self.Q[state, action] = self.Q[state, action] + self.alpha * (
                    self.R[state, action] + self.gamma * max_Q_next - self.Q[state, action]
                )

                # Move to next state
                state = next_state
                available_act = self.available_actions(state)

                # Check if episode is done (no available actions)
                if available_act.size == 0:
                    break

                action = self.sample_next_action(available_act)

            # Print progress
            if verbose and (episode + 1) % 200 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed")

        if verbose:
            print("Training completed!")

    def get_normalized_q_table(self):
        """
        Get Q-table normalized to 0-100 scale for visualization.

        Returns:
            Normalized Q-table
        """
        if self.Q.max() > 0:
            return (self.Q / self.Q.max() * 100).astype(int)
        return self.Q.astype(int)

    def get_optimal_path(self, start_state, goal_state):
        """
        Get optimal path from start to goal using learned Q-values.

        Args:
            start_state: Starting state
            goal_state: Goal state

        Returns:
            List of states in the optimal path
        """
        path = [start_state]
        current_state = start_state
        visited = set([start_state])

        # Prevent infinite loops
        max_steps = self.num_states * 2
        steps = 0

        while current_state != goal_state and steps < max_steps:
            # Choose action with highest Q-value
            next_state = np.argmax(self.Q[current_state])

            # Check for loops
            if next_state in visited:
                print(f"Warning: Loop detected at state {next_state}")
                break

            path.append(next_state)
            visited.add(next_state)
            current_state = next_state
            steps += 1

        return path

    def visualize_q_table(self, save_path='./q_table_heatmap.png'):
        """
        Visualize the learned Q-table as a heatmap.

        Args:
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.Q, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Q-Value')
        plt.title('Learned Q-Table Heatmap')
        plt.xlabel('Action (Next State)')
        plt.ylabel('State')

        # Add text annotations
        for i in range(self.num_states):
            for j in range(self.num_states):
                text = plt.text(j, i, f'{self.Q[i, j]:.0f}',
                               ha="center", va="center", color="white", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Q-table visualization saved to {save_path}")
        plt.close()


def main():
    """
    Main function to demonstrate Q-Learning.
    """
    # Define the reward matrix
    # Rows: current state, Columns: action (which leads to next state)
    # -1: not allowed, 0: allowed with no reward, 100: goal state
    R = np.array([
        [ -1, -1, -1, -1,  0, -1 ],  # State 0: can go to state 4
        [ -1, -1, -1,  0, -1, 100 ],  # State 1: can go to states 3 or 5 (goal)
        [ -1, -1, -1,  0, -1, -1 ],  # State 2: can go to state 3
        [ -1,  0,  0, -1,  0, -1 ],  # State 3: can go to states 1, 2, or 4
        [  0, -1, -1,  0, -1, 100 ],  # State 4: can go to states 0 or 5 (goal)
        [ -1,  0, -1, -1,  0, 100 ]   # State 5: goal state (self-loop)
    ])

    print("=" * 60)
    print("Simple Q-Learning Example")
    print("=" * 60)
    print("\nReward Matrix:")
    print(R)
    print("\nGoal: Learn to reach state 5 from any starting state")
    print("-" * 60)

    # Create and train agent
    agent = SimpleQLearning(reward_matrix=R, gamma=0.8, alpha=0.9)
    agent.train(num_episodes=1000, verbose=True)

    # Display results
    print("\n" + "=" * 60)
    print("Learned Q-Table (normalized to 0-100):")
    print("=" * 60)
    Q_normalized = agent.get_normalized_q_table()
    print(Q_normalized)

    # Test optimal path
    print("\n" + "=" * 60)
    print("Testing Optimal Paths:")
    print("=" * 60)

    for start_state in range(5):  # Test from states 0-4
        path = agent.get_optimal_path(start_state, goal_state=5)
        print(f"State {start_state} -> {' -> '.join(map(str, path))}")

    # Visualize Q-table
    agent.visualize_q_table()

    print("\n" + "=" * 60)
    print("Q-Learning completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
