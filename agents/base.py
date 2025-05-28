import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Neural network for policy
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
    @abstractmethod
    def select_action(self, state):
        """Select an action based on the current state."""
        pass
    
    @abstractmethod
    def update(self, experience):
        """Update the agent's policy based on experience."""
        pass
    
    def save_model(self, path):
        """Save the agent's model."""
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        """Load the agent's model."""
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 