import torch
import torch.nn as nn
import numpy as np
from .base import BaseAgent

class StudentAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        super().__init__(state_dim, action_dim, learning_rate)
        
        # Additional network for curriculum understanding
        self.curriculum_understanding = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
        
        self.curriculum_optimizer = torch.optim.Adam(self.curriculum_understanding.parameters(), lr=learning_rate)
        
    def select_action(self, state):
        """Select an action based on the current state and curriculum."""
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_logits = self.policy_network(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
        return action
    
    def understand_curriculum(self, state, curriculum):
        """Process and understand the curriculum provided by the teacher."""
        state_curriculum = torch.cat([torch.FloatTensor(state), torch.FloatTensor(curriculum)])
        with torch.no_grad():
            understanding = self.curriculum_understanding(state_curriculum)
        return understanding.numpy()
    
    def update(self, experience):
        """Update the student agent's policy based on experience."""
        states, actions, rewards, next_states, curricula = experience
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        curricula = torch.FloatTensor(curricula)
        
        # Update policy network
        action_logits = self.policy_network(states)
        action_probs = torch.softmax(action_logits, dim=-1)
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1))
        
        policy_loss = -torch.mean(torch.log(selected_action_probs) * rewards)
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Update curriculum understanding
        curriculum_loss = self._update_curriculum_understanding(states, curricula, rewards)
        
        return {
            'policy_loss': policy_loss.item(),
            'curriculum_loss': curriculum_loss
        }
    
    def _update_curriculum_understanding(self, states, curricula, rewards):
        """Update the curriculum understanding network."""
        state_curricula = torch.cat([states, curricula], dim=1)
        understanding_pred = self.curriculum_understanding(state_curricula)
        
        # Curriculum understanding loss based on how well the agent follows the curriculum
        curriculum_loss = torch.mean((understanding_pred - states) ** 2) * rewards.mean()
        
        self.curriculum_optimizer.zero_grad()
        curriculum_loss.backward()
        self.curriculum_optimizer.step()
        
        return curriculum_loss.item()
    
    def get_learning_progress(self):
        """Get the current learning progress of the student."""
        return {
            'policy_network': self.policy_network.state_dict(),
            'curriculum_understanding': self.curriculum_understanding.state_dict()
        } 