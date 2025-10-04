"""
Prioritized Experience Replay Buffer for MHA-DQN
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer with importance sampling.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Beta increment per sampling step
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        # Storage
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        logger.info(f"Prioritized replay buffer initialized with capacity {capacity}")
    
    def add(self, experience: Dict[str, Any]):
        """Add experience to buffer."""
        # Calculate priority (use max priority for new experiences)
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample batch with prioritized sampling."""
        if self.size < batch_size:
            raise ValueError(f"Buffer size {self.size} < batch size {batch_size}")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normalize by max weight
        
        # Extract experiences
        batch = {
            'market_data': [],
            'sentiment_data': [],
            'actions': [],
            'rewards': [],
            'next_market_data': [],
            'next_sentiment_data': [],
            'dones': [],
            'indices': indices,
            'weights': weights
        }
        
        for idx in indices:
            experience = self.buffer[idx]
            batch['market_data'].append(experience['market_data'])
            batch['sentiment_data'].append(experience['sentiment_data'])
            batch['actions'].append(experience['action'])
            batch['rewards'].append(experience['reward'])
            batch['next_market_data'].append(experience['next_market_data'])
            batch['next_sentiment_data'].append(experience['next_sentiment_data'])
            batch['dones'].append(experience['done'])
        
        # Convert to numpy arrays
        batch['market_data'] = np.array(batch['market_data'])
        batch['sentiment_data'] = np.array(batch['sentiment_data'])
        batch['actions'] = np.array(batch['actions'])
        batch['rewards'] = np.array(batch['rewards'])
        batch['next_market_data'] = np.array(batch['next_market_data'])
        batch['next_sentiment_data'] = np.array(batch['next_sentiment_data'])
        batch['dones'] = np.array(batch['dones'])
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + 1e-6  # Add small epsilon to avoid zero priorities
        self.priorities[indices] = priorities
    
    def __len__(self):
        return self.size


class SimpleReplayBuffer:
    """
    Simple experience replay buffer (uniform sampling).
    """
    
    def __init__(self, capacity: int):
        """Initialize simple replay buffer."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
        logger.info(f"Simple replay buffer initialized with capacity {capacity}")
    
    def add(self, experience: Dict[str, Any]):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample batch uniformly."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer size {len(self.buffer)} < batch size {batch_size}")
        
        # Sample experiences
        experiences = random.sample(self.buffer, batch_size)
        
        # Extract batch
        batch = {
            'market_data': [],
            'sentiment_data': [],
            'actions': [],
            'rewards': [],
            'next_market_data': [],
            'next_sentiment_data': [],
            'dones': []
        }
        
        for experience in experiences:
            batch['market_data'].append(experience['market_data'])
            batch['sentiment_data'].append(experience['sentiment_data'])
            batch['actions'].append(experience['action'])
            batch['rewards'].append(experience['reward'])
            batch['next_market_data'].append(experience['next_market_data'])
            batch['next_sentiment_data'].append(experience['next_sentiment_data'])
            batch['dones'].append(experience['done'])
        
        # Convert to numpy arrays
        batch['market_data'] = np.array(batch['market_data'])
        batch['sentiment_data'] = np.array(batch['sentiment_data'])
        batch['actions'] = np.array(batch['actions'])
        batch['rewards'] = np.array(batch['rewards'])
        batch['next_market_data'] = np.array(batch['next_market_data'])
        batch['next_sentiment_data'] = np.array(batch['next_sentiment_data'])
        batch['dones'] = np.array(batch['dones'])
        
        return batch
    
    def __len__(self):
        return len(self.buffer)
