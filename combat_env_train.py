from combat_env import CombatEnv
from models.dqn import DeepQNetwork
from models.qn import QLearningNetwork
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import os
import pickle
from abc import ABC, abstractmethod

# Initialize the environment
env = CombatEnv()

# Initialize the agent
agent = DeepQNetwork(
    context_dim=6,  # State has 6 elements after discretization
    action_dim=5,   # 5 possible actions
    config={},
    model_file='model_weight/dqn_model.pth'
)
model_name = 'dqn_model'

agent = QLearningNetwork(
    context_dim=5,  # State has 6 elements after discretization
    action_dim=5,   # 5 possible actions
    config={},
    model_file='model_weight/qn_model.pth'
)
model_name = 'qn_model'


agent.epsilon = 0  # Set epsilon to 0 for evaluation (greedy policy)

# Define the evaluation function with a print_output flag
def evaluate_agent(agent, print_output=False):
    test_cases = [
        {
            'description': 'Boss is attacking, freeze skill in cd, player should dodge.',
            'state': (1, 1, 0, 1, 80, 100),  # boss_state=1, freeze_skill_cd=0, consecutive_attack_count=0, medicine_count=1+, player_health_bin=2 (health between 40-60)
            'expected_action': 2  # Dodge
        },
        {
            'description': 'Boss is frozen, player should attack.',
            'state': (2, 1, 0, 1, 80, 100),  # boss_state=2, freeze_skill_cd=1, consecutive_attack_count=0, medicine_count=1+, player_health_bin=4 (health between 80-100)
            'expected_action': 1  # Attack
        },
        {
            'description': 'Player has low health, boss not attacking, should heal.',
            'state': (0, 1, 0, 1, 20, 100),  # boss_state=0, freeze_skill_cd=0, consecutive_attack_count=0, medicine_count=1+, player_health_bin=1 (health between 20-40)
            'expected_action': 4  # Use medicine
        },
        {
            'description': 'Player has high health, boss not attacking, should not heal.',
            'state': (0, 1, 0, 1, 80, 100),  # boss_state=0, freeze_skill_cd=0, consecutive_attack_count=0, medicine_count=1+, player_health_bin=4 (health between 80-100)
            'expected_action': [0, 1, 2, 3]  # Do not medicine
        },
        {
            'description': 'Player has full health, should not use medicine.',
            'state': (0, 1, 0, 1, 100, 100),  # player_health_bin=5 (health 100)
            'expected_action': [0, 1, 2, 3]  # Not use medicine
        },
        {
            'description': 'Freeze skill on cooldown, should not try to freeze.',
            'state': (1, 1, 0, 1, 60, 100),  # freeze_skill_cd=1
            'expected_action': 2  # Dodge (since boss is attacking)
        },
        {
            'description': 'No medicines left, should not try to heal.',
            'state': (0, 0, 0, 0, 40, 100),  # medicine_count=0
            'expected_action': [0, 1, 2, 3]   # Not use medicine
        },
        {
            'description': 'Boss not attacking, player should attack.',
            'state': (0, 0, 0, 1, 100, 100),
            'expected_action': 1  # Attack
        },
        {
            'description': 'Boss attacking, freeze skill ready, should freeze.',
            'state': (1, 0, 0, 1, 80, 100),
            'expected_action': 3  # Use freeze skill
        },
        {
            'description': 'Boss attacking, low health, should dodge over attack.',
            'state': (1, 1, 0, 1, 20, 100),
            'expected_action': 2  # Dodge
        },
    ]

    success_count = 0
    total_tests = len(test_cases)

    if print_output:
        print("Evaluation Results:")
        print("-------------------")

    for idx, test_case in enumerate(test_cases):
        state = test_case['state']
        expected_action = test_case['expected_action']
        description = test_case['description']

        # Agent chooses action
        state = agent.discretize_state(state)
        action = agent.choose_action(state)

        # Check if the action matches the expected action
        if isinstance(expected_action, int):
            success = action == expected_action
        else:
            success = action in expected_action
        
        success_str = "PASS" if success else "FAIL"

        if print_output:
            # Print the results
            print(f"Test Case {idx+1}: {description}")
            print(f"Expected Action: {expected_action}, Agent's Action: {action} -> {success_str}")
            print()

        if success:
            success_count += 1

    # Calculate success rate
    success_rate = (success_count / total_tests) * 100

    if print_output:
        print(f"Overall Success Rate: {success_rate}%")

    return success_rate

# Run the evaluation without printing output
evaluate_agent(agent, print_output=True)


num_episodes = 20000
initial_log_step = num_episodes//100
initial_log_step_cap = 20000//20
agent.epsilon_decay = (agent.epsilon - agent.epsilon_min) / num_episodes
total_rewards = []
player_health = []
boss_health = []
boss_defeated = []
agent.save_model(f"model_weight/qn_model_{0}.pth")
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        # Discretize the state
        state_discrete = agent.discretize_state(state)
        # Agent chooses action
        action = agent.choose_action(state_discrete)
        # Take action in the environment
        next_state, reward, done, info = env.step(action)
        # Discretize next_state
        next_state_discrete = agent.discretize_state(next_state)
        # Store experience
        agent.store_data(state_discrete, action, reward, next_state_discrete, done)
        # Train the agent
        # agent.train_network()
        # Update total reward
        total_reward += reward
        # Move to next state
        state = next_state
        if done:
            player_health.append(state[4])
            boss_health.append(state[5])
            boss_defeated.append(state[5] <= 0)
    # Append total reward
    total_rewards.append(total_reward)
    # Optionally print progress
    if (episode + 1) % (num_episodes // 10) == 0 or ((episode + 1) <= initial_log_step_cap and (episode + 1) % initial_log_step == 0):
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Evaluation Success Rate: {evaluate_agent(agent)}")
        agent.save_model(f"model_weight/{model_name}_{episode + 1}.pth")

# Save the trained model
agent.save_model()

# Parameters for moving average
window_size = 100  # Adjust based on your preference

# Compute the moving average
moving_avg = np.convolve(total_rewards, np.ones(window_size)/window_size, mode='valid')

# Create episode ranges
episodes = np.arange(1, num_episodes + 1)
moving_avg_episodes = np.arange(window_size, num_episodes + 1)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(episodes, total_rewards, label='Total Reward per Episode', alpha=0.5)
plt.plot(moving_avg_episodes, moving_avg, label=f'Moving Average (window size = {window_size})', color='red')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards and Moving Average per Episode')
plt.legend()
plt.grid(True)
plt.show()

# Compute moving averages for player health, boss health, and boss defeated rate
player_health_avg = np.convolve(player_health, np.ones(window_size) / window_size, mode='valid')
boss_health_avg = np.convolve(boss_health, np.ones(window_size) / window_size, mode='valid')
boss_defeated_avg = np.convolve(boss_defeated, np.ones(window_size) / window_size, mode='valid') * 100

# Create episode range for health and defeated rate plots
health_avg_episodes = np.arange(window_size, num_episodes + 1)

# Plotting average player and boss health, and boss defeated rate
plt.figure(figsize=(12, 6))
plt.plot(health_avg_episodes, player_health_avg, label='Average Player Health (window size = 100)', color='blue')
plt.plot(health_avg_episodes, boss_health_avg, label='Average Boss Health (window size = 100)', color='green')
plt.plot(health_avg_episodes, boss_defeated_avg, label='Boss Defeated Rate (window size = 100)', color='orange')
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Average Player & Boss Health and Boss Defeated Rate (Moving Average)')
plt.legend()
plt.grid(True)
plt.show()