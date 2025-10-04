#!/usr/bin/env python3
"""
Training Script for MHA-DQN Portfolio Optimization Research
Trains the Multi-Head Attention Deep Q-Network model

Usage:
    python scripts/train_mha_dqn.py --config configs/mha_dqn_config.yaml
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.mha_dqn import MHADQNAgent, MHADQNNetwork
from training.environment import PortfolioEnvironment
from training.replay_buffer import PrioritizedReplayBuffer
from utils.config import load_config, get_device_config
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


class MHADQNTrainer:
    """
    Comprehensive trainer for MHA-DQN portfolio optimization.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = get_device_config()
        
        # Data directories
        self.data_dir = Path(config.get('data_dir', './data'))
        self.features_dir = self.data_dir / 'features'
        self.models_dir = Path('./models')
        self.results_dir = Path('./results')
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.num_episodes = config['training']['num_episodes']
        self.max_steps_per_episode = config['training']['max_steps_per_episode']
        self.batch_size = config['training']['batch_size']
        
        # Initialize components
        self.env = None
        self.agent = None
        self.replay_buffer = None
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_losses': [],
            'episode_lengths': [],
            'portfolio_values': [],
            'sharpe_ratios': [],
            'max_drawdowns': []
        }
        
        logger.info(f"MHADQNTrainer initialized on device: {self.device}")
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load processed feature data."""
        logger.info("Loading processed feature data")
        
        feature_files = list(self.features_dir.glob("*_normalized.csv"))
        
        if not feature_files:
            raise FileNotFoundError(f"No normalized feature files found in {self.features_dir}")
        
        data = {}
        expected_features = 30  # Expected number of features (excluding date)
        
        for file in feature_files:
            symbol = file.stem.replace('_normalized', '')
            try:
                df = pd.read_csv(file, index_col=0, parse_dates=True)
                df = df.sort_index()
                
                # Skip BRK.B as it has incomplete fundamental data
                if symbol == 'BRK.B':
                    logger.warning(f"Skipping {symbol} due to incomplete fundamental data ({len(df.columns)} features vs expected {expected_features})")
                    continue
                
                # Verify feature count consistency
                if len(df.columns) != expected_features:
                    logger.warning(f"Skipping {symbol} due to feature count mismatch ({len(df.columns)} vs expected {expected_features})")
                    continue
                
                data[symbol] = df
                logger.info(f"Loaded {len(df)} samples for {symbol} with {len(df.columns)} features")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        logger.info(f"Loaded data for {len(data)} symbols")
        return data
    
    def prepare_data_for_training(self, data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, List[str]]:
        """Prepare data for MHA-DQN training."""
        logger.info("Preparing data for training")
        
        # Get common date range
        all_dates = None
        for symbol, df in data.items():
            if all_dates is None:
                all_dates = df.index
            else:
                all_dates = all_dates.intersection(df.index)
        
        logger.info(f"Common date range: {len(all_dates)} days from {all_dates.min()} to {all_dates.max()}")
        
        # Align all data to common dates
        aligned_data = {}
        for symbol, df in data.items():
            aligned_data[symbol] = df.loc[all_dates]
        
        # Create combined feature matrix
        symbols = list(aligned_data.keys())
        n_dates = len(all_dates)
        n_features_per_symbol = len(aligned_data[symbols[0]].columns)
        n_symbols = len(symbols)
        
        # Combined feature matrix: (n_dates, n_symbols * n_features)
        combined_features = np.zeros((n_dates, n_symbols * n_features_per_symbol))
        
        for i, symbol in enumerate(symbols):
            start_idx = i * n_features_per_symbol
            end_idx = (i + 1) * n_features_per_symbol
            combined_features[:, start_idx:end_idx] = aligned_data[symbol].values
        
        # Handle NaN values
        combined_features = np.nan_to_num(combined_features, nan=0.0)
        
        logger.info(f"Prepared feature matrix: {combined_features.shape}")
        logger.info(f"Features per symbol: {n_features_per_symbol}")
        logger.info(f"Total features: {n_symbols * n_features_per_symbol}")
        
        return combined_features, symbols
    
    def create_environment(self, features: np.ndarray, symbols: List[str]) -> PortfolioEnvironment:
        """Create portfolio environment."""
        logger.info("Creating portfolio environment")
        
        env_config = self.config['environment']
        
        env = PortfolioEnvironment(
            features=features,
            symbols=symbols,
            initial_capital=env_config['portfolio']['initial_capital'],
            transaction_cost=env_config['portfolio']['transaction_cost'],
            max_position_size=env_config['portfolio']['max_position_size'],
            lookback_window=env_config['state']['lookback_window'],
            reward_config=env_config['reward']
        )
        
        logger.info(f"Environment created with {len(symbols)} assets")
        return env
    
    def create_agent(self, env: PortfolioEnvironment) -> MHADQNAgent:
        """Create MHA-DQN agent."""
        logger.info("Creating MHA-DQN agent")
        
        # Update config with environment info
        model_config = self.config.copy()
        model_config['num_stocks'] = len(env.symbols)
        model_config['market_features'] = env.n_total_features // env.n_symbols  # Features per stock
        model_config['sentiment_features'] = 10  # 10 sentiment features per stock
        model_config['seq_len'] = env.lookback_window
        
        # Ensure the config has the required structure for MHADQNNetwork
        if 'attention' not in model_config:
            model_config['attention'] = self.config['model']['attention']
        if 'network' not in model_config:
            model_config['network'] = self.config['model']['network']
        
        agent = MHADQNAgent(model_config, device=self.device)
        
        logger.info(f"Agent created with {agent.q_network.count_parameters():,} parameters")
        return agent
    
    def create_replay_buffer(self) -> PrioritizedReplayBuffer:
        """Create prioritized replay buffer."""
        logger.info("Creating prioritized replay buffer")
        
        buffer_config = self.config['training']['replay_buffer']
        
        buffer = PrioritizedReplayBuffer(
            capacity=buffer_config['capacity'],
            alpha=buffer_config['alpha'],
            beta=buffer_config['beta'],
            beta_increment=buffer_config['beta_increment']
        )
        
        logger.info(f"Replay buffer created with capacity {buffer_config['capacity']}")
        return buffer
    
    def train_episode(self, episode: int) -> Dict[str, float]:
        """Train one episode."""
        try:
            state = self.env.reset()
            episode_reward = 0.0
            episode_loss = 0.0
            episode_length = 0
            
            # Convert state to tensors
            market_data = torch.FloatTensor(state['market_features']).unsqueeze(0).to(self.device)
            sentiment_data = torch.FloatTensor(state['sentiment_features']).unsqueeze(0).to(self.device)
        
            for step in range(self.max_steps_per_episode):  # Use full episode length for champion model
                # Select action
                with torch.no_grad():
                    action = self.agent.select_action(market_data, sentiment_data, training=True)
                    action_np = action.cpu().numpy().flatten()
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action_np)
                
                # Convert next state to tensors
                next_market_data = torch.FloatTensor(next_state['market_features']).unsqueeze(0).to(self.device)
                next_sentiment_data = torch.FloatTensor(next_state['sentiment_features']).unsqueeze(0).to(self.device)
                
                # Store experience in replay buffer
                experience = {
                    'market_data': state['market_features'],
                    'sentiment_data': state['sentiment_features'],
                    'action': action_np,
                    'reward': reward,
                    'next_market_data': next_state['market_features'],
                    'next_sentiment_data': next_state['sentiment_features'],
                    'done': done
                }
                
                self.replay_buffer.add(experience)
                
                # Train agent if buffer has enough samples
                if len(self.replay_buffer) > self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    
                    # Convert numpy arrays to PyTorch tensors
                    batch['market_data'] = torch.FloatTensor(batch['market_data']).to(self.device)
                    batch['sentiment_data'] = torch.FloatTensor(batch['sentiment_data']).to(self.device)
                    batch['actions'] = torch.FloatTensor(batch['actions']).to(self.device)
                    batch['rewards'] = torch.FloatTensor(batch['rewards']).to(self.device)
                    batch['next_market_data'] = torch.FloatTensor(batch['next_market_data']).to(self.device)
                    batch['next_sentiment_data'] = torch.FloatTensor(batch['next_sentiment_data']).to(self.device)
                    batch['dones'] = torch.FloatTensor(batch['dones']).to(self.device)
                    
                    loss_info = self.agent.train_step(batch)
                    episode_loss += loss_info['loss']
                
                # Update state
                state = next_state
                market_data = next_market_data
                sentiment_data = next_sentiment_data
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Calculate episode metrics
            portfolio_value = self.env.portfolio_value
            returns = np.array(self.env.portfolio_returns)
            
            # Calculate Sharpe ratio
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            portfolio_values = np.array(self.env.portfolio_value_history)
            if len(portfolio_values) > 1:
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values - peak) / peak
                max_drawdown = np.min(drawdown)
            else:
                max_drawdown = 0.0
            
            return {
                'episode_rewards': episode_reward,
                'episode_losses': episode_loss / max(episode_length, 1),
                'episode_lengths': episode_length,
                'portfolio_values': portfolio_value,
                'sharpe_ratios': sharpe_ratio,
                'max_drawdowns': max_drawdown
            }
        except Exception as e:
            logger.error(f"Error in episode {episode}: {e}")
            # Return default values if episode fails
            return {
                'episode_rewards': 0.0,
                'episode_losses': 0.0,
                'episode_lengths': 0,
                'portfolio_values': self.env.initial_capital,
                'sharpe_ratios': 0.0,
                'max_drawdowns': 0.0
            }
    
    def train(self) -> Dict[str, List[float]]:
        """Run complete training loop."""
        logger.info(f"Starting training for {self.num_episodes} episodes")
        
        # Load and prepare data
        data = self.load_data()
        features, symbols = self.prepare_data_for_training(data)
        
        # Create environment and agent
        self.env = self.create_environment(features, symbols)
        self.agent = self.create_agent(self.env)
        self.replay_buffer = self.create_replay_buffer()
        
        # Training loop
        best_sharpe = -np.inf
        
        for episode in tqdm(range(self.num_episodes), desc="Training Episodes"):
            # Train episode
            episode_metrics = self.train_episode(episode)
            
            # Store metrics
            for key, value in episode_metrics.items():
                self.training_metrics[key].append(value)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_metrics['episode_rewards'][-100:])
                avg_sharpe = np.mean(self.training_metrics['sharpe_ratios'][-100:])
                avg_loss = np.mean(self.training_metrics['episode_losses'][-100:])
                
                logger.info(f"Episode {episode}: Avg Reward={avg_reward:.4f}, "
                          f"Avg Sharpe={avg_sharpe:.4f}, Avg Loss={avg_loss:.4f}")
            
            # Save best model
            current_sharpe = episode_metrics['sharpe_ratios']
            if current_sharpe > best_sharpe:
                best_sharpe = current_sharpe
                self.save_model(f"best_model_episode_{episode}.pth")
        
        logger.info("Training completed!")
        return self.training_metrics
    
    def save_model(self, filename: str):
        """Save model checkpoint."""
        filepath = self.models_dir / filename
        self.agent.save_checkpoint(str(filepath))
        logger.info(f"Model saved: {filepath}")
    
    def save_training_metrics(self):
        """Save training metrics to file."""
        metrics_file = self.results_dir / "training_metrics.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, values in self.training_metrics.items():
            serializable_metrics[key] = [float(v) if not np.isnan(v) else 0.0 for v in values]
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info(f"Training metrics saved: {metrics_file}")
    
    def plot_training_progress(self):
        """Plot training progress."""
        logger.info("Creating training progress plots")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MHA-DQN Training Progress', fontsize=16, fontweight='bold')
        
        # Episode rewards
        axes[0, 0].plot(self.training_metrics['episode_rewards'], alpha=0.7)
        axes[0, 0].plot(pd.Series(self.training_metrics['episode_rewards']).rolling(100).mean(), 
                       linewidth=2, color='red', label='100-episode MA')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode losses
        axes[0, 1].plot(self.training_metrics['episode_losses'], alpha=0.7)
        axes[0, 1].plot(pd.Series(self.training_metrics['episode_losses']).rolling(100).mean(), 
                       linewidth=2, color='red', label='100-episode MA')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Portfolio values
        axes[0, 2].plot(self.training_metrics['portfolio_values'], alpha=0.7)
        axes[0, 2].plot(pd.Series(self.training_metrics['portfolio_values']).rolling(100).mean(), 
                       linewidth=2, color='red', label='100-episode MA')
        axes[0, 2].set_title('Portfolio Value')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Value ($)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Sharpe ratios
        axes[1, 0].plot(self.training_metrics['sharpe_ratios'], alpha=0.7)
        axes[1, 0].plot(pd.Series(self.training_metrics['sharpe_ratios']).rolling(100).mean(), 
                       linewidth=2, color='red', label='100-episode MA')
        axes[1, 0].set_title('Sharpe Ratio')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Max drawdowns
        axes[1, 1].plot(self.training_metrics['max_drawdowns'], alpha=0.7)
        axes[1, 1].plot(pd.Series(self.training_metrics['max_drawdowns']).rolling(100).mean(), 
                       linewidth=2, color='red', label='100-episode MA')
        axes[1, 1].set_title('Maximum Drawdown')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Max Drawdown')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[1, 2].plot(self.training_metrics['episode_lengths'], alpha=0.7)
        axes[1, 2].plot(pd.Series(self.training_metrics['episode_lengths']).rolling(100).mean(), 
                       linewidth=2, color='red', label='100-episode MA')
        axes[1, 2].set_title('Episode Length')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Steps')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / "training_progress.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Training progress plot saved: {plot_file}")
    
    def print_training_summary(self):
        """Print training summary."""
        logger.info("Training Summary:")
        logger.info("=" * 50)
        
        final_metrics = {
            'Final Episode Reward': self.training_metrics['episode_rewards'][-1],
            'Average Episode Reward': np.mean(self.training_metrics['episode_rewards']),
            'Best Episode Reward': np.max(self.training_metrics['episode_rewards']),
            'Final Sharpe Ratio': self.training_metrics['sharpe_ratios'][-1],
            'Average Sharpe Ratio': np.mean(self.training_metrics['sharpe_ratios']),
            'Best Sharpe Ratio': np.max(self.training_metrics['sharpe_ratios']),
            'Final Portfolio Value': self.training_metrics['portfolio_values'][-1],
            'Average Training Loss': np.mean(self.training_metrics['episode_losses']),
            'Final Max Drawdown': self.training_metrics['max_drawdowns'][-1],
            'Average Max Drawdown': np.mean(self.training_metrics['max_drawdowns'])
        }
        
        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info("=" * 50)


def main():
    """Main function for training script."""
    parser = argparse.ArgumentParser(description='Train MHA-DQN for portfolio optimization')
    parser.add_argument('--config', type=str, default='configs/mha_dqn_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--episodes', type=int, help='Number of training episodes (overrides config)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save training progress plots')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override episodes if specified
    if args.episodes:
        config['training']['num_episodes'] = args.episodes
    
    # Initialize trainer
    trainer = MHADQNTrainer(config)
    
    # Run training
    try:
        logger.info("Starting MHA-DQN training...")
        training_metrics = trainer.train()
        
        # Save results
        trainer.save_training_metrics()
        trainer.save_model("final_model.pth")
        
        # Create plots
        if args.save_plots:
            trainer.plot_training_progress()
        
        # Print summary
        trainer.print_training_summary()
        
        print("\n" + "="*50)
        print("MHA-DQN TRAINING COMPLETED")
        print("="*50)
        print(f"Episodes trained: {len(training_metrics['episode_rewards'])}")
        print(f"Final Sharpe ratio: {training_metrics['sharpe_ratios'][-1]:.4f}")
        print(f"Best Sharpe ratio: {np.max(training_metrics['sharpe_ratios']):.4f}")
        print(f"Final portfolio value: ${training_metrics['portfolio_values'][-1]:,.2f}")
        print(f"Models saved to: {trainer.models_dir}")
        print(f"Results saved to: {trainer.results_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
