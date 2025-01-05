import numpy as np
import random
import pickle
import os
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from models.base_agent import BaseAgent

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128]):
        super(QNetwork, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DeepQNetwork(BaseAgent):
    def __init__(self, context_dim, action_dim, config, model_file):
        super(DeepQNetwork, self).__init__(config, model_file)
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.model_file = model_file

        # Initialize Q-network and target network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(context_dim, action_dim).to(self.device)
        self.target_network = QNetwork(context_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer and loss function
        self.lr = config.get('lr', 1e-3)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 1e-5)
        self.epsilon_min = config.get('epsilon_min', 0.1)
        self.batch_size = config.get('batch_size', 64)
        self.replay_buffer_size = config.get('replay_buffer_size', 10000)
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.update_every = config.get('update_every', 1000)
        self.step_count = 0
        self.target_update_freq = config.get('target_update_freq', 1000)

        # Logging
        self.loss_history = []
        self.q_history = []

        # Load model if exists
        self.load_model()

        # Define state normalization parameters
        self.state_min = np.array([0, 0, 0, 0, 0, 0])       # [boss_state, freeze_skill_cd, consecutive_attack_count, medicine_count, player_health, boss_health]
        self.state_max = np.array([2, 1, 5, 10, 100, 100])

    def normalize_state(self, state):
        """
        Normalize the state to a [0, 1] range based on predefined min and max values.
        """
        state = np.array(state, dtype=np.float32)
        normalized = (state - self.state_min) / (self.state_max - self.state_min)
        normalized = np.clip(normalized, 0, 1)  # Ensure values are within [0, 1]
        return normalized

    def choose_action(self, state):
        """
        Choose an action based on the current state using an epsilon-greedy policy.
        """
        state = self.normalize_state(state)  # Normalize the state
        if random.random() <= self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        # Exponential decay for epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= 0.995  # Decay rate
            self.epsilon = max(self.epsilon, self.epsilon_min)
        return action

    def store_data(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer and initiate training.
        """
        state = self.normalize_state(state)
        next_state = self.normalize_state(next_state)
        scaled_reward = reward / 1000.0  # Scale reward to reduce magnitude
        self.replay_buffer.append((state, action, scaled_reward, next_state, done))
        self.step_count += 1

        # Learn every step if enough samples are available
        if len(self.replay_buffer) >= self.batch_size:
            self.train_network()

        # Update target network periodically
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

    def train_network(self):
        """
        Sample a batch from replay buffer and perform a training step.
        """
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        self.loss_history.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Track average Q-values
        avg_q = current_q.mean().item()
        self.q_history.append(avg_q)

    def save_model(self, file_path = None):
        file_path = file_path or self.model_file
        """
        Save the model's state to a file.
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'q_history': self.q_history
        }, file_path)

    def load_model(self):
        """
        Load the model's state from a file if it exists.
        """
        if os.path.exists(self.model_file):
            checkpoint = torch.load(self.model_file, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.loss_history = checkpoint.get('loss_history', [])
            self.q_history = checkpoint.get('q_history', [])
            print(f"Model loaded from {self.model_file}")
        else:
            print("No existing model found. Initializing a new model.")

    def update_target_network(self):
        """
        Update the target network with the Q-network's weights.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
