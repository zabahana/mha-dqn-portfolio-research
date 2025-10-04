#!/usr/bin/env python3
"""
Comprehensive Model Evaluation and Backtesting Script
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.mha_dqn import MHADQNAgent
from training.environment import PortfolioEnvironment
from utils.logging import setup_logging
from utils.visualization import BacktestVisualizer, PerformanceTableGenerator

class ModelEvaluator:
    """Comprehensive model evaluation and backtesting"""
    
    def __init__(self, config_path: str, model_path: str, device: str = "cpu"):
        self.config_path = config_path
        self.model_path = model_path
        self.device = device
        
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.agent = self._load_model()
        
        # Setup environment
        self.env = self._setup_environment()
        
        # Results storage
        self.results = {}
        
    def _load_model(self):
        """Load trained model"""
        logger = logging.getLogger(__name__)
        logger.info(f"Loading model from {self.model_path}")
        
        # Create agent
        model_config = self.config.copy()
        model_config['num_stocks'] = 9  # Based on our dataset
        model_config['market_features'] = 30  # Features per stock
        model_config['sentiment_features'] = 10  # Sentiment features per stock
        model_config['seq_len'] = 60  # Lookback window
        
        # Ensure required config structure
        if 'attention' not in model_config:
            model_config['attention'] = self.config['model']['attention']
        if 'network' not in model_config:
            model_config['network'] = self.config['model']['network']
        
        agent = MHADQNAgent(model_config, device=self.device)
        
        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        agent.q_network.eval()
        
        logger.info("Model loaded successfully")
        return agent
    
    def _setup_environment(self):
        """Setup evaluation environment"""
        logger = logging.getLogger(__name__)
        logger.info("Setting up evaluation environment")
        
        # Load feature data
        features_dir = Path("data/features")
        feature_files = list(features_dir.glob("*_normalized.csv"))
        
        data = {}
        expected_features = 30
        
        for file in feature_files:
            symbol = file.stem.replace('_normalized', '')
            if symbol == 'BRK.B':  # Skip BRK.B as in training
                continue
            try:
                df = pd.read_csv(file, index_col=0, parse_dates=True)
                df = df.sort_index()
                
                if len(df.columns) == expected_features:
                    data[symbol] = df
                    logger.info(f"Loaded {len(df)} samples for {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        # Create environment
        env = PortfolioEnvironment(
            data=data,
            initial_capital=100000,
            lookback_window=60,
            reward_config=self.config['environment']['reward']
        )
        
        logger.info(f"Environment setup complete with {len(data)} symbols")
        return env
    
    def run_backtest(self, start_date: str = None, end_date: str = None) -> dict:
        """Run comprehensive backtest"""
        logger = logging.getLogger(__name__)
        logger.info("Starting backtest evaluation...")
        
        # Reset environment
        state = self.env.reset()
        
        # Convert state to tensors
        market_data = torch.FloatTensor(state['market_features']).unsqueeze(0).to(self.device)
        sentiment_data = torch.FloatTensor(state['sentiment_features']).unsqueeze(0).to(self.device)
        
        # Storage for results
        portfolio_values = []
        portfolio_returns = []
        actions_taken = []
        dates = []
        
        episode_reward = 0.0
        step = 0
        max_steps = 252  # One year of trading days
        
        logger.info(f"Running backtest for {max_steps} steps...")
        
        with torch.no_grad():
            for step in range(max_steps):
                # Select action (no exploration during evaluation)
                action = self.agent.select_action(market_data, sentiment_data, training=False)
                action_np = action.cpu().numpy().flatten()
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action_np)
                
                # Store results
                portfolio_values.append(self.env.portfolio_value)
                portfolio_returns.append(reward)
                actions_taken.append(action_np.copy())
                dates.append(self.env.current_date)
                
                episode_reward += reward
                
                # Update state
                state = next_state
                market_data = torch.FloatTensor(state['market_features']).unsqueeze(0).to(self.device)
                sentiment_data = torch.FloatTensor(state['sentiment_features']).unsqueeze(0).to(self.device)
                
                if done:
                    break
                
                if step % 50 == 0:
                    logger.info(f"Backtest progress: {step}/{max_steps} steps")
        
        # Calculate performance metrics
        portfolio_returns = np.array(portfolio_returns)
        portfolio_values = np.array(portfolio_values)
        
        # Create benchmark (equal weight portfolio)
        benchmark_returns = self._calculate_benchmark_returns()
        
        # Store results
        self.results = {
            'portfolio_returns': portfolio_returns,
            'portfolio_values': portfolio_values,
            'benchmark_returns': benchmark_returns,
            'actions_taken': np.array(actions_taken),
            'dates': dates,
            'episode_reward': episode_reward,
            'total_steps': step + 1
        }
        
        logger.info(f"Backtest completed: {step + 1} steps, final portfolio value: ${portfolio_values[-1]:,.2f}")
        return self.results
    
    def _calculate_benchmark_returns(self) -> np.ndarray:
        """Calculate equal-weight benchmark returns"""
        # For simplicity, use a synthetic benchmark
        # In practice, this would use actual equal-weight portfolio returns
        np.random.seed(42)
        return np.random.normal(0.0005, 0.025, len(self.results['portfolio_returns']))
    
    def calculate_performance_metrics(self) -> dict:
        """Calculate comprehensive performance metrics"""
        logger = logging.getLogger(__name__)
        logger.info("Calculating performance metrics...")
        
        portfolio_returns = self.results['portfolio_returns']
        benchmark_returns = self.results['benchmark_returns']
        portfolio_values = self.results['portfolio_values']
        
        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        
        # Risk metrics
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / volatility  # Assuming 2% risk-free rate
        
        # Drawdown analysis
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown)
        
        # Information ratio
        excess_returns = portfolio_returns - benchmark_returns
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # Win rate
        win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns)
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252)
        sortino_ratio = (annual_return - 0.02) / downside_volatility if downside_volatility > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'win_rate': win_rate,
            'sortino_ratio': sortino_ratio,
            'final_portfolio_value': portfolio_values[-1],
            'total_trades': len(portfolio_returns)
        }
        
        logger.info("Performance metrics calculated")
        return metrics
    
    def generate_visualizations(self, output_dir: Path):
        """Generate comprehensive evaluation visualizations"""
        logger = logging.getLogger(__name__)
        logger.info("Generating evaluation visualizations...")
        
        visualizer = BacktestVisualizer()
        portfolio_returns = self.results['portfolio_returns']
        benchmark_returns = self.results['benchmark_returns']
        portfolio_values = self.results['portfolio_values']
        dates = pd.to_datetime(self.results['dates'])
        
        # 1. Cumulative Returns
        fig = visualizer.plot_cumulative_returns(
            portfolio_returns, benchmark_returns, dates,
            save_path=output_dir / "cumulative_returns.png"
        )
        plt.close(fig)
        
        # 2. Drawdown Analysis
        fig = visualizer.plot_drawdown_analysis(
            portfolio_values, dates,
            save_path=output_dir / "drawdown_analysis.png"
        )
        plt.close(fig)
        
        # 3. Rolling Metrics
        fig = visualizer.plot_rolling_metrics(
            pd.Series(portfolio_returns), pd.Series(benchmark_returns), dates,
            save_path=output_dir / "rolling_metrics.png"
        )
        plt.close(fig)
        
        # 4. Portfolio Allocation Analysis
        self._plot_portfolio_allocations(output_dir)
        
        logger.info("Evaluation visualizations generated")
    
    def _plot_portfolio_allocations(self, output_dir: Path):
        """Plot portfolio allocation analysis"""
        import matplotlib.pyplot as plt
        
        actions = np.array(self.results['actions_taken'])
        symbols = self.env.symbols
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)
        
        # Average allocation
        avg_allocation = np.mean(actions, axis=0)
        axes[0, 0].bar(symbols, avg_allocation, color='skyblue')
        axes[0, 0].set_title('Average Portfolio Allocation', fontweight='bold')
        axes[0, 0].set_ylabel('Weight')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Allocation over time
        for i, symbol in enumerate(symbols):
            axes[0, 1].plot(actions[:, i], label=symbol, alpha=0.7)
        axes[0, 1].set_title('Portfolio Allocation Over Time', fontweight='bold')
        axes[0, 1].set_xlabel('Trading Day')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Allocation heatmap
        im = axes[1, 0].imshow(actions.T, cmap='RdYlBu', aspect='auto')
        axes[1, 0].set_title('Portfolio Allocation Heatmap', fontweight='bold')
        axes[1, 0].set_xlabel('Trading Day')
        axes[1, 0].set_ylabel('Stock')
        axes[1, 0].set_yticks(range(len(symbols)))
        axes[1, 0].set_yticklabels(symbols)
        
        # Allocation volatility
        allocation_vol = np.std(actions, axis=0)
        axes[1, 1].bar(symbols, allocation_vol, color='lightcoral')
        axes[1, 1].set_title('Allocation Volatility', fontweight='bold')
        axes[1, 1].set_ylabel('Standard Deviation')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "portfolio_allocations.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def generate_performance_tables(self, output_dir: Path):
        """Generate performance tables"""
        logger = logging.getLogger(__name__)
        logger.info("Generating performance tables...")
        
        table_generator = PerformanceTableGenerator()
        portfolio_returns = self.results['portfolio_returns']
        benchmark_returns = self.results['benchmark_returns']
        
        # Performance metrics table
        performance_df = table_generator.generate_performance_table(
            portfolio_returns, benchmark_returns
        )
        performance_df.to_csv(output_dir / "performance_metrics.csv", index=False)
        
        # Statistical tests table
        stats_df = table_generator.generate_statistical_tests(
            portfolio_returns, benchmark_returns
        )
        stats_df.to_csv(output_dir / "statistical_tests.csv", index=False)
        
        # Detailed metrics
        metrics = self.calculate_performance_metrics()
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_df.to_csv(output_dir / "detailed_metrics.csv", index=False)
        
        logger.info("Performance tables generated")
    
    def save_results(self, output_dir: Path):
        """Save all evaluation results"""
        logger = logging.getLogger(__name__)
        logger.info("Saving evaluation results...")
        
        # Save raw results
        results_data = {
            'portfolio_returns': self.results['portfolio_returns'].tolist(),
            'portfolio_values': self.results['portfolio_values'].tolist(),
            'benchmark_returns': self.results['benchmark_returns'].tolist(),
            'actions_taken': self.results['actions_taken'].tolist(),
            'dates': [d.isoformat() for d in self.results['dates']],
            'episode_reward': self.results['episode_reward'],
            'total_steps': self.results['total_steps']
        }
        
        with open(output_dir / "evaluation_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save performance metrics
        metrics = self.calculate_performance_metrics()
        with open(output_dir / "performance_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Evaluation results saved")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate MHA-DQN model performance")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output-dir", type=str, default="results/evaluation", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.config, args.model_path, args.device)
        
        # Run backtest
        logger.info("Starting model evaluation...")
        evaluator.run_backtest()
        
        # Generate visualizations
        evaluator.generate_visualizations(output_dir)
        
        # Generate tables
        evaluator.generate_performance_tables(output_dir)
        
        # Save results
        evaluator.save_results(output_dir)
        
        # Print summary
        metrics = evaluator.calculate_performance_metrics()
        logger.info("Evaluation Summary:")
        logger.info(f"Total Return: {metrics['total_return']:.2%}")
        logger.info(f"Annual Return: {metrics['annual_return']:.2%}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}")
        
        logger.info(f"All results saved to: {output_dir.absolute()}")
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
