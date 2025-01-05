# models/ddqn_resnet.py
# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from models.base_agent import BaseAgent
from torchvision import models
from torchvision.models import ResNet18_Weights

class QNetwork(nn.Module):
    def __init__(self, action_dim, context_dim):
        super(QNetwork, self).__init__()
        # Load the pre-trained ResNet18 model
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Freeze all parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Unfreeze the last two layers (layer4 and fc)
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        # Modify the last layer of ResNet18 to output feature vectors instead of classification results
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        # Concatenate ResNet features with context features
        self.fc1 = nn.Linear(num_ftrs + context_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.head = nn.Linear(256, action_dim)

    def forward(self, x, context_features):
        x = self.resnet(x)
        x = torch.cat((x, context_features), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.head(x)
        return x

class DDQN_RESNET(BaseAgent):
    def __init__(self, state_dim, action_dim, context_dim, config, model_file):
        super(DDQN_RESNET, self).__init__(config, model_file)
        self.action_dim = action_dim
        self.context_dim = context_dim
        # Create the network
        self.policy_net = QNetwork(self.action_dim, context_dim)
        self.target_net = QNetwork(self.action_dim, context_dim)
        self.update_target_network()
        # Define the optimizer and loss function
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.policy_net.parameters()), lr=self.lr)
        self.criterion = nn.MSELoss()
        # Load the model (if it exists)
        self.load_model()
        # Set the device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def choose_action(self, state):
        state_image, context_features = state
        state_image = torch.from_numpy(state_image).float().to(self.device)
        context_features = torch.from_numpy(np.array(context_features)).float().to(self.device).unsqueeze(0)
        if random.random() <= self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_value = self.policy_net(state_image, context_features)
                action = q_value.max(1)[1].item()
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        return action

    def store_data(self, state, action, reward, next_state, done):
        state_image, context_features = state
        next_state_image, next_context_features = next_state
        state_image = np.squeeze(state_image, axis=0)
        next_state_image = np.squeeze(next_state_image, axis=0)
        self.replay_buffer.append((state_image, context_features, action, reward, next_state_image, next_context_features, done))

    def train_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_images = np.array([data[0] for data in minibatch])
        state_contexts = np.array([data[1] for data in minibatch])
        action_batch = np.array([data[2] for data in minibatch])
        reward_batch = np.array([data[3] for data in minibatch])
        next_state_images = np.array([data[4] for data in minibatch])
        next_state_contexts = np.array([data[5] for data in minibatch])
        done_batch = np.array([data[6] for data in minibatch])

        state_images = torch.from_numpy(state_images).float().to(self.device)
        state_contexts = torch.from_numpy(state_contexts).float().to(self.device)
        action_batch = torch.from_numpy(action_batch).long().to(self.device).unsqueeze(1)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        next_state_images = torch.from_numpy(next_state_images).float().to(self.device)
        next_state_contexts = torch.from_numpy(next_state_contexts).float().to(self.device)
        done_batch = torch.from_numpy(done_batch).float().to(self.device)

        # Calculate current Q values
        q_values = self.policy_net(state_images, state_contexts).gather(1, action_batch)
        # Use DDQN to update and calculate next Q values
        with torch.no_grad():
            next_state_actions = self.policy_net(next_state_images, next_state_contexts).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_images, next_state_contexts).gather(1, next_state_actions).squeeze()
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Calculate loss
        loss = self.criterion(q_values.squeeze(), target_q_values)

        # Update network parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
