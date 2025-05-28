import torch
import torch.nn as nn
import numpy as np
from .base import BaseAgent

class TeachingAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, curriculum_dim, learning_rate=0.001):
        super().__init__(state_dim, action_dim, learning_rate)
        self.curriculum_dim = curriculum_dim
        
        # Additional network for curriculum generation
        self.curriculum_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, curriculum_dim)
        )
        
        # Network for evaluating student performance
        self.evaluation_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.curriculum_optimizer = torch.optim.Adam(self.curriculum_network.parameters(), lr=learning_rate)
        self.evaluation_optimizer = torch.optim.Adam(self.evaluation_network.parameters(), lr=learning_rate)
        
    def select_action(self, state):
        """Select a teaching action based on the current state."""
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_logits = self.policy_network(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
        return action
    
    def generate_curriculum(self, state, action):
        """Generate a curriculum based on current state and action."""
        state_action = torch.cat([torch.FloatTensor(state), torch.FloatTensor([action])])
        with torch.no_grad():
            curriculum = self.curriculum_network(state_action)
        return curriculum.numpy()
    
    def evaluate_student(self, state, action):
        """Evaluate student performance."""
        state_action = torch.cat([torch.FloatTensor(state), torch.FloatTensor([action])])
        with torch.no_grad():
            evaluation = self.evaluation_network(state_action)
        return evaluation.item()
    
    def update(self, experience):
        """Update the teaching agent's policy based on experience."""
        states, actions, rewards, next_states = experience
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        
        # Update policy network
        action_logits = self.policy_network(states)
        action_probs = torch.softmax(action_logits, dim=-1)
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1))
        
        policy_loss = -torch.mean(torch.log(selected_action_probs) * rewards)
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Update curriculum network
        curriculum_loss = self._update_curriculum(states, actions, rewards)
        
        # Update evaluation network
        evaluation_loss = self._update_evaluation(states, actions, rewards)
        
        return {
            'policy_loss': policy_loss.item(),
            'curriculum_loss': curriculum_loss,
            'evaluation_loss': evaluation_loss
        }
    
    def _update_curriculum(self, states, actions, rewards):
        """Update the curriculum generation network."""
        state_actions = torch.cat([states, actions.float().unsqueeze(1)], dim=1)
        curriculum_pred = self.curriculum_network(state_actions)
        
        # Simple curriculum loss based on student performance
        curriculum_loss = torch.mean((curriculum_pred - rewards.unsqueeze(1)) ** 2)
        
        self.curriculum_optimizer.zero_grad()
        curriculum_loss.backward()
        self.curriculum_optimizer.step()
        
        return curriculum_loss.item()
    
    def _update_evaluation(self, states, actions, rewards):
        """Update the student evaluation network."""
        state_actions = torch.cat([states, actions.float().unsqueeze(1)], dim=1)
        evaluation_pred = self.evaluation_network(state_actions)
        
        evaluation_loss = torch.mean((evaluation_pred - rewards.unsqueeze(1)) ** 2)
        
        self.evaluation_optimizer.zero_grad()
        evaluation_loss.backward()
        self.evaluation_optimizer.step()
        
        return evaluation_loss.item() 