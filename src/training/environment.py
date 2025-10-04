"""
Portfolio Environment for MHA-DQN Training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PortfolioEnvironment:
    """
    Portfolio trading environment for reinforcement learning.
    """
    
    def __init__(self, 
                 features: np.ndarray,
                 symbols: List[str],
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 0.2,
                 lookback_window: int = 60,
                 reward_config: Dict = None):
        """
        Initialize portfolio environment.
        
        Args:
            features: Feature matrix (n_timesteps, n_features)
            symbols: List of stock symbols
            initial_capital: Initial portfolio capital
            transaction_cost: Transaction cost rate
            max_position_size: Maximum position size per asset
            lookback_window: Number of historical steps for state
            reward_config: Reward function configuration
        """
        self.features = features
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        
        # Reward configuration
        self.reward_config = reward_config or {
            'return_weight': 1.0,
            'risk_penalty': 0.5,
            'sentiment_bonus': 0.3,
            'transaction_penalty': 0.1
        }
        
        # Environment state
        self.n_timesteps, self.n_total_features = features.shape
        self.n_symbols = len(symbols)
        self.n_features_per_stock = self.n_total_features // self.n_symbols
        
        # Portfolio state
        self.current_step = 0
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = np.zeros(self.n_symbols)  # Number of shares
        self.portfolio_weights = np.zeros(self.n_symbols)  # Portfolio weights
        
        # History tracking
        self.portfolio_value_history = [initial_capital]
        self.portfolio_returns = []
        self.action_history = []
        self.reward_history = []
        
        logger.info(f"Portfolio environment initialized with {self.n_symbols} assets, "
                   f"{self.n_timesteps} timesteps, {self.n_features_per_stock} features per stock")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state."""
        self.current_step = self.lookback_window
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = np.zeros(self.n_symbols)
        self.portfolio_weights = np.zeros(self.n_symbols)
        
        # Clear history
        self.portfolio_value_history = [self.initial_capital]
        self.portfolio_returns = []
        self.action_history = []
        self.reward_history = []
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Portfolio weights (should sum to <= 1.0)
            
        Returns:
            next_state: Next state observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        # Ensure action is valid
        action = np.clip(action, 0, self.max_position_size)
        action = action / (np.sum(action) + 1e-8)  # Normalize to sum to 1
        
        # Calculate portfolio return
        if self.current_step < self.n_timesteps - 1:
            # Get current and next prices (using close price feature)
            current_prices = self._get_prices(self.current_step)
            next_prices = self._get_prices(self.current_step + 1)
            
            # Calculate returns
            price_returns = (next_prices - current_prices) / (current_prices + 1e-8)
            
            # Calculate portfolio return
            portfolio_return = np.sum(self.portfolio_weights * price_returns)
            
            # Update portfolio value
            self.portfolio_value *= (1 + portfolio_return)
            
            # Calculate transaction costs
            weight_changes = np.abs(action - self.portfolio_weights)
            transaction_costs = np.sum(weight_changes) * self.transaction_cost * self.portfolio_value
            self.portfolio_value -= transaction_costs
            
            # Update positions
            self.portfolio_weights = action.copy()
            
            # Store history
            self.portfolio_value_history.append(self.portfolio_value)
            self.portfolio_returns.append(portfolio_return)
            self.action_history.append(action.copy())
            
            # Move to next step
            self.current_step += 1
            
            # Calculate reward
            reward = self._calculate_reward(portfolio_return, weight_changes, action)
            self.reward_history.append(reward)
            
            # Check if done
            done = self.current_step >= self.n_timesteps - 1
            
            # Additional info
            info = {
                'portfolio_value': self.portfolio_value,
                'portfolio_return': portfolio_return,
                'transaction_costs': transaction_costs,
                'portfolio_weights': self.portfolio_weights.copy()
            }
            
        else:
            # Episode finished
            reward = 0.0
            done = True
            info = {'portfolio_value': self.portfolio_value}
        
        next_state = self._get_state()
        
        return next_state, reward, done, info
    
    def _get_state(self) -> Dict[str, np.ndarray]:
        """Get current state observation."""
        if self.current_step < self.lookback_window:
            # Pad with zeros if not enough history
            start_idx = 0
            pad_length = self.lookback_window - self.current_step
            historical_features = self.features[start_idx:self.current_step + 1]
        else:
            start_idx = self.current_step - self.lookback_window + 1
            pad_length = 0
            historical_features = self.features[start_idx:self.current_step + 1]
        
        # Pad if necessary to ensure exactly lookback_window length
        if pad_length > 0:
            padding = np.zeros((pad_length, self.n_total_features))
            historical_features = np.vstack([padding, historical_features])
        
        # Ensure exactly lookback_window length
        if historical_features.shape[0] > self.lookback_window:
            historical_features = historical_features[-self.lookback_window:]
        
        # Reshape to separate market and sentiment features
        # Each stock has 34 features: 7 basic + 24 fundamental + 3 sentiment
        # We'll use all features as market features and create dummy sentiment features
        market_features = historical_features  # Use all 306 features as market features
        
        # Create dummy sentiment features (10 features per stock as expected by the model)
        n_sentiment_features = self.n_symbols * 10
        sentiment_features = np.zeros((historical_features.shape[0], n_sentiment_features))
        
        return {
            'market_features': market_features,
            'sentiment_features': sentiment_features
        }
    
    def _get_prices(self, step: int) -> np.ndarray:
        """Extract price information from features."""
        # Assume the first feature for each stock is the close price
        prices = []
        for i in range(self.n_symbols):
            start_idx = i * self.n_features_per_stock
            # Use the first feature as price (this is a simplification)
            price = self.features[step, start_idx]
            prices.append(price)
        
        return np.array(prices)
    
    def _calculate_reward(self, portfolio_return: float, weight_changes: np.ndarray, action: np.ndarray) -> float:
        """Calculate reward for current step."""
        # Base return reward
        return_reward = portfolio_return * self.reward_config['return_weight']
        
        # Risk penalty (volatility of recent returns)
        if len(self.portfolio_returns) >= 10:
            recent_returns = np.array(self.portfolio_returns[-10:])
            volatility = np.std(recent_returns)
            risk_penalty = volatility * self.reward_config['risk_penalty']
        else:
            risk_penalty = 0.0
        
        # Transaction cost penalty
        transaction_penalty = np.sum(weight_changes) * self.reward_config['transaction_penalty']
        
        # Diversification reward (entropy of weights)
        weights = action + 1e-8  # Add small epsilon to avoid log(0)
        weights = weights / np.sum(weights)
        diversification_reward = -np.sum(weights * np.log(weights)) * self.reward_config.get('diversification_weight', 0.1)
        
        total_reward = return_reward + risk_penalty + transaction_penalty + diversification_reward
        
        return total_reward
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        if len(self.portfolio_returns) < 2:
            return {}
        
        returns = np.array(self.portfolio_returns)
        values = np.array(self.portfolio_value_history)
        
        # Total return
        total_return = (values[-1] - values[0]) / values[0]
        
        # Annualized return (assuming daily data)
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'final_value': values[-1]
        }
