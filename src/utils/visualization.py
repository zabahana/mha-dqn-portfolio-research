"""
Comprehensive visualization utilities for MHA-DQN Portfolio Optimization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PortfolioVisualizer:
    """Comprehensive visualization class for portfolio analysis"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
    def plot_price_series(self, data: Dict[str, pd.DataFrame], 
                         symbols: List[str], 
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot normalized price series for multiple stocks"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=self.dpi)
        axes = axes.flatten()
        
        for i, symbol in enumerate(symbols[:9]):
            if symbol in data:
                df = data[symbol]
                if 'close' in df.columns:
                    # Normalize to starting value
                    normalized_prices = df['close'] / df['close'].iloc[0]
                    axes[i].plot(df.index, normalized_prices, color=self.colors[i % len(self.colors)], linewidth=2)
                    axes[i].set_title(f'{symbol} Normalized Price', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel('Date')
                    axes[i].set_ylabel('Normalized Price')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(symbols), 9):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def plot_correlation_matrix(self, data: Dict[str, pd.DataFrame], 
                               symbols: List[str], 
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot correlation matrix of stock returns"""
        # Calculate returns
        returns_data = {}
        for symbol in symbols:
            if symbol in data and 'close' in data[symbol].columns:
                returns = data[symbol]['close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        if not returns_data:
            raise ValueError("No valid return data found")
            
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        ax.set_title('Stock Returns Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def plot_volatility_analysis(self, data: Dict[str, pd.DataFrame], 
                                symbols: List[str], 
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot rolling volatility analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=self.dpi)
        
        # Rolling volatility
        for i, symbol in enumerate(symbols):
            if symbol in data and 'close' in data[symbol].columns:
                returns = data[symbol]['close'].pct_change().dropna()
                rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
                # Ensure same length for x and y
                valid_indices = rolling_vol.dropna().index
                valid_vol = rolling_vol.dropna()
                ax1.plot(valid_indices, valid_vol, 
                        label=symbol, color=self.colors[i % len(self.colors)], alpha=0.7)
        
        ax1.set_title('30-Day Rolling Volatility (Annualized)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Volatility')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volatility distribution
        volatilities = []
        labels = []
        for symbol in symbols:
            if symbol in data and 'close' in data[symbol].columns:
                returns = data[symbol]['close'].pct_change().dropna()
                vol = returns.std() * np.sqrt(252)
                volatilities.append(vol)
                labels.append(symbol)
        
        ax2.bar(labels, volatilities, color=self.colors[:len(labels)])
        ax2.set_title('Annualized Volatility by Stock', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Stock Symbol')
        ax2.set_ylabel('Volatility')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def plot_feature_distribution(self, features_data: Dict[str, pd.DataFrame], 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Plot distribution of engineered features"""
        # Sample a few key features for visualization
        sample_features = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'volume_sma_ratio']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=self.dpi)
        axes = axes.flatten()
        
        for i, feature in enumerate(sample_features[:6]):
            if i < len(axes):
                feature_values = []
                for symbol, df in features_data.items():
                    if feature in df.columns:
                        feature_values.extend(df[feature].dropna().values)
                
                if feature_values:
                    axes[i].hist(feature_values, bins=50, alpha=0.7, color=self.colors[i % len(self.colors)])
                    axes[i].set_title(f'{feature.upper()} Distribution', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(sample_features), 6):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig

class ModelVisualizer:
    """Visualization utilities for model architecture and training"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_model_architecture(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create model architecture diagram"""
        fig, ax = plt.subplots(figsize=(14, 10), dpi=self.dpi)
        
        # Define components
        components = [
            ("Market Features\n(270 features)", (1, 8), (2, 2)),
            ("Sentiment Features\n(90 features)", (1, 6), (2, 2)),
            ("Multi-Head\nAttention", (4, 7), (3, 2)),
            ("Temporal\nEncoding", (4, 5), (3, 2)),
            ("Cross-Attention\nFusion", (7, 6), (3, 2)),
            ("Dueling DQN\nNetwork", (10, 6), (3, 2)),
            ("Q-Values\n(9 stocks)", (13, 6), (2, 2)),
        ]
        
        # Draw components
        for text, (x, y), (w, h) in components:
            rect = plt.Rectangle((x, y), w, h, facecolor='lightblue', 
                               edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # Draw arrows
        arrows = [
            ((2, 7), (4, 7)),  # Market to Attention
            ((2, 6), (4, 6)),  # Sentiment to Attention
            ((5, 7), (7, 7)),  # Attention to Cross-Attention
            ((5, 5), (7, 5)),  # Encoding to Cross-Attention
            ((8, 6), (10, 6)), # Cross-Attention to DQN
            ((11, 6), (13, 6)), # DQN to Q-Values
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        ax.set_xlim(0, 16)
        ax.set_ylim(3, 10)
        ax.set_title('MHA-DQN Architecture', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def plot_training_progress(self, metrics: Dict[str, List], 
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot comprehensive training progress"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        
        # Episode Rewards
        if 'episode_rewards' in metrics:
            axes[0, 0].plot(metrics['episode_rewards'], color='blue', alpha=0.7)
            axes[0, 0].set_title('Episode Rewards', fontweight='bold')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode Losses
        if 'episode_losses' in metrics:
            axes[0, 1].plot(metrics['episode_losses'], color='red', alpha=0.7)
            axes[0, 1].set_title('Episode Losses', fontweight='bold')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Sharpe Ratios
        if 'sharpe_ratios' in metrics:
            axes[0, 2].plot(metrics['sharpe_ratios'], color='green', alpha=0.7)
            axes[0, 2].set_title('Sharpe Ratios', fontweight='bold')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Sharpe Ratio')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Portfolio Values
        if 'portfolio_values' in metrics:
            axes[1, 0].plot(metrics['portfolio_values'], color='purple', alpha=0.7)
            axes[1, 0].set_title('Portfolio Values', fontweight='bold')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Portfolio Value ($)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Max Drawdowns
        if 'max_drawdowns' in metrics:
            axes[1, 1].plot(metrics['max_drawdowns'], color='orange', alpha=0.7)
            axes[1, 1].set_title('Max Drawdowns', fontweight='bold')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Max Drawdown')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Episode Lengths
        if 'episode_lengths' in metrics:
            axes[1, 2].plot(metrics['episode_lengths'], color='brown', alpha=0.7)
            axes[1, 2].set_title('Episode Lengths', fontweight='bold')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Steps')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def plot_attention_weights(self, attention_weights: np.ndarray, 
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot attention weight heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        ax.set_title('Attention Weights Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig

class BacktestVisualizer:
    """Visualization utilities for backtesting and performance analysis"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_cumulative_returns(self, portfolio_returns: np.ndarray, 
                               benchmark_returns: np.ndarray,
                               dates: pd.DatetimeIndex,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot cumulative returns comparison"""
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        
        # Calculate cumulative returns
        portfolio_cumret = np.cumprod(1 + portfolio_returns) - 1
        benchmark_cumret = np.cumprod(1 + benchmark_returns) - 1
        
        ax.plot(dates, portfolio_cumret * 100, label='MHA-DQN Portfolio', 
               linewidth=2, color='blue')
        ax.plot(dates, benchmark_cumret * 100, label='Benchmark (Equal Weight)', 
               linewidth=2, color='red', linestyle='--')
        
        ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def plot_drawdown_analysis(self, portfolio_values: np.ndarray,
                              dates: pd.DatetimeIndex,
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot drawdown analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=self.dpi)
        
        # Portfolio value
        ax1.plot(dates, portfolio_values, color='blue', linewidth=2)
        ax1.set_title('Portfolio Value Over Time', fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100
        
        ax2.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(dates, drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown Analysis', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig
    
    def plot_rolling_metrics(self, portfolio_returns: np.ndarray,
                           benchmark_returns: np.ndarray,
                           dates: pd.DatetimeIndex,
                           window: int = 30,
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot rolling performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        # Rolling Sharpe Ratio
        portfolio_rolling_sharpe = portfolio_returns.rolling(window).mean() / portfolio_returns.rolling(window).std() * np.sqrt(252)
        benchmark_rolling_sharpe = benchmark_returns.rolling(window).mean() / benchmark_returns.rolling(window).std() * np.sqrt(252)
        
        axes[0, 0].plot(dates[window:], portfolio_rolling_sharpe[window:], label='MHA-DQN', color='blue')
        axes[0, 0].plot(dates[window:], benchmark_rolling_sharpe[window:], label='Benchmark', color='red')
        axes[0, 0].set_title(f'{window}-Day Rolling Sharpe Ratio', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rolling Volatility
        portfolio_rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252) * 100
        benchmark_rolling_vol = benchmark_returns.rolling(window).std() * np.sqrt(252) * 100
        
        axes[0, 1].plot(dates[window:], portfolio_rolling_vol[window:], label='MHA-DQN', color='blue')
        axes[0, 1].plot(dates[window:], benchmark_rolling_vol[window:], label='Benchmark', color='red')
        axes[0, 1].set_title(f'{window}-Day Rolling Volatility', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling Returns
        portfolio_rolling_ret = portfolio_returns.rolling(window).mean() * 252 * 100
        benchmark_rolling_ret = benchmark_returns.rolling(window).mean() * 252 * 100
        
        axes[1, 0].plot(dates[window:], portfolio_rolling_ret[window:], label='MHA-DQN', color='blue')
        axes[1, 0].plot(dates[window:], benchmark_rolling_ret[window:], label='Benchmark', color='red')
        axes[1, 0].set_title(f'{window}-Day Rolling Annualized Returns', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling Beta
        rolling_beta = portfolio_returns.rolling(window).cov(benchmark_returns) / benchmark_returns.rolling(window).var()
        axes[1, 1].plot(dates[window:], rolling_beta[window:], color='green')
        axes[1, 1].set_title(f'{window}-Day Rolling Beta', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        return fig

class PerformanceTableGenerator:
    """Generate performance tables and statistical summaries"""
    
    def __init__(self):
        pass
    
    def generate_performance_table(self, portfolio_returns: np.ndarray,
                                 benchmark_returns: np.ndarray,
                                 risk_free_rate: float = 0.02) -> pd.DataFrame:
        """Generate comprehensive performance metrics table"""
        
        # Calculate metrics
        portfolio_annual_return = np.mean(portfolio_returns) * 252
        benchmark_annual_return = np.mean(benchmark_returns) * 252
        
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
        benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252)
        
        portfolio_sharpe = (portfolio_annual_return - risk_free_rate) / portfolio_volatility
        benchmark_sharpe = (benchmark_annual_return - risk_free_rate) / benchmark_volatility
        
        # Maximum drawdown
        portfolio_cumret = np.cumprod(1 + portfolio_returns)
        portfolio_peak = np.maximum.accumulate(portfolio_cumret)
        portfolio_drawdown = (portfolio_cumret - portfolio_peak) / portfolio_peak
        portfolio_max_dd = np.min(portfolio_drawdown)
        
        benchmark_cumret = np.cumprod(1 + benchmark_returns)
        benchmark_peak = np.maximum.accumulate(benchmark_cumret)
        benchmark_drawdown = (benchmark_cumret - benchmark_peak) / benchmark_peak
        benchmark_max_dd = np.min(benchmark_drawdown)
        
        # Calmar ratio
        portfolio_calmar = portfolio_annual_return / abs(portfolio_max_dd)
        benchmark_calmar = benchmark_annual_return / abs(benchmark_max_dd)
        
        # Information ratio
        excess_returns = portfolio_returns - benchmark_returns
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # Create DataFrame
        metrics = {
            'Metric': [
                'Annual Return (%)',
                'Annual Volatility (%)',
                'Sharpe Ratio',
                'Maximum Drawdown (%)',
                'Calmar Ratio',
                'Information Ratio'
            ],
            'MHA-DQN Portfolio': [
                f"{portfolio_annual_return * 100:.2f}",
                f"{portfolio_volatility * 100:.2f}",
                f"{portfolio_sharpe:.3f}",
                f"{portfolio_max_dd * 100:.2f}",
                f"{portfolio_calmar:.3f}",
                f"{information_ratio:.3f}"
            ],
            'Benchmark (Equal Weight)': [
                f"{benchmark_annual_return * 100:.2f}",
                f"{benchmark_volatility * 100:.2f}",
                f"{benchmark_sharpe:.3f}",
                f"{benchmark_max_dd * 100:.2f}",
                f"{benchmark_calmar:.3f}",
                "N/A"
            ]
        }
        
        return pd.DataFrame(metrics)
    
    def generate_statistical_tests(self, portfolio_returns: np.ndarray,
                                 benchmark_returns: np.ndarray) -> pd.DataFrame:
        """Generate statistical significance tests"""
        from scipy import stats
        
        # T-test for returns
        t_stat, p_value = stats.ttest_ind(portfolio_returns, benchmark_returns)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.ks_2samp(portfolio_returns, benchmark_returns)
        
        # Mann-Whitney U test
        u_stat, u_p_value = stats.mannwhitneyu(portfolio_returns, benchmark_returns, alternative='two-sided')
        
        tests = {
            'Test': [
                'T-test (Returns)',
                'Kolmogorov-Smirnov Test',
                'Mann-Whitney U Test'
            ],
            'Statistic': [
                f"{t_stat:.4f}",
                f"{ks_stat:.4f}",
                f"{u_stat:.4f}"
            ],
            'P-value': [
                f"{p_value:.4f}",
                f"{ks_p_value:.4f}",
                f"{u_p_value:.4f}"
            ],
            'Significant (α=0.05)': [
                "Yes" if p_value < 0.05 else "No",
                "Yes" if ks_p_value < 0.05 else "No",
                "Yes" if u_p_value < 0.05 else "No"
            ]
        }
        
        return pd.DataFrame(tests)
