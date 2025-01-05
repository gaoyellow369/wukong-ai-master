import numpy as np
import random
import pickle
import os
from models.base_agent import BaseAgent

class QLearningNetwork(BaseAgent):
    def __init__(self, context_dim, action_dim, config, model_file):
        super(QLearningNetwork, self).__init__(config, model_file)
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.model_file = model_file
        # Initialize Q-table
        self.q_table = {}
        # Load model if exists
        self.load_model()
        # Set parameters
        self.lr = config.get('lr', 0.01)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.01)
        self.epsilon_min = config.get('epsilon_min', 0.1)
        self.batch_size = config.get('batch_size', 32)
        self.replay_buffer = []

    def choose_action(self, state):
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if random.random() <= self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            action = np.argmax(self.q_table[state_key])
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        return action

    def store_data(self, state, action, reward, next_state, done):
        #self.train_network(state, action, reward, next_state, done)
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)
        # Q-learning update
        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        target = reward + self.gamma * next_max * (1 - int(done))
        self.q_table[state_key][action] = old_value + self.lr * (target - old_value)

    def train_network(self):
        pass

    def save_model(self, file_path = None):
        file_path = file_path or self.model_file
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                self.q_table = pickle.load(f)
        else:
            self.q_table = {}
    def update_target_network(self):
            pass 
    
    def discretize_state(self, state):
        boss_state = state[0]  # 0, 1, or 2
        freeze_skill_cd = state[1]  # 0 or 1
        consecutive_attack_count = state[2]  # 0 to 5
        medicine_count = state[3]  # 0 to 5
        player_health = state[4]  # 0 to 100
        boss_health = state[5]  # 0 to 100
        # Discretize health into bins of size 10
        player_health_bin = int(player_health // 10)
        boss_health_bin = int(boss_health // 20)
        return (boss_state, freeze_skill_cd, consecutive_attack_count, min(1, medicine_count),
            player_health_bin)