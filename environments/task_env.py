import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TeachingTaskEnv(gym.Env):
    def __init__(self, task_complexity=1.0):
        super().__init__()
        
        self.task_complexity = task_complexity
        self.state_dim = 10
        self.action_dim = 5
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        self.state = None
        self.steps = 0
        self.max_steps = 100
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = np.random.randn(self.state_dim)
        self.steps = 0
        return self.state, {}
    
    def step(self, action):
        self.steps += 1
        
        # Simulate task progression
        next_state = self.state + np.random.randn(self.state_dim) * 0.1
        
        # Calculate reward based on action and task complexity
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = self.steps >= self.max_steps
        
        # Additional info
        info = {
            'task_complexity': self.task_complexity,
            'steps': self.steps
        }
        
        self.state = next_state
        return next_state, reward, done, False, info
    
    def _calculate_reward(self, action):
        """Calculate reward based on action and task complexity."""
        # Base reward for taking any action
        reward = 0.1
        
        # Add complexity-based reward
        if action == 0:  # Correct action for simple tasks
            reward += 1.0 * (1.0 - self.task_complexity)
        elif action == 1:  # Correct action for medium tasks
            reward += 1.0 * (1.0 - abs(0.5 - self.task_complexity))
        elif action == 2:  # Correct action for complex tasks
            reward += 1.0 * self.task_complexity
        
        # Add noise to make learning more challenging
        reward += np.random.randn() * 0.1
        
        return reward
    
    def get_task_difficulty(self):
        """Return the current task difficulty."""
        return self.task_complexity
    
    def set_task_difficulty(self, difficulty):
        """Set the task difficulty."""
        self.task_complexity = np.clip(difficulty, 0.0, 1.0) 