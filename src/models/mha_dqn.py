"""
Multi-Head Attention Deep Q-Network for Portfolio Optimization

This module implements the core MHA-DQN architecture with:
- Multi-head self-attention for temporal modeling
- Cross-attention for sentiment integration
- Dueling network architecture
- Residual connections and layer normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import math
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequences.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with optional cross-attention.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor (batch_size, seq_len_q, d_model)
            key: Key tensor (batch_size, seq_len_k, d_model)
            value: Value tensor (batch_size, seq_len_v, d_model)
            mask: Attention mask (batch_size, seq_len_q, seq_len_k)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        
        return output, attention_weights.mean(dim=1)  # Average attention across heads


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and feed-forward network.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class SentimentFusionModule(nn.Module):
    """
    Module for fusing sentiment information with market features using cross-attention.
    """
    
    def __init__(self, market_dim: int, sentiment_dim: int, fusion_dim: int, 
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.market_dim = market_dim
        self.sentiment_dim = sentiment_dim
        self.fusion_dim = fusion_dim
        
        # Project inputs to fusion dimension
        self.market_proj = nn.Linear(market_dim, fusion_dim)
        self.sentiment_proj = nn.Linear(sentiment_dim, fusion_dim)
        
        # Cross-attention layers
        self.market_to_sentiment = MultiHeadAttention(fusion_dim, num_heads, dropout)
        self.sentiment_to_market = MultiHeadAttention(fusion_dim, num_heads, dropout)
        
        # Fusion layers
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_ff = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, market_features: torch.Tensor, 
                sentiment_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_features: Market features (batch_size, seq_len, market_dim)
            sentiment_features: Sentiment features (batch_size, seq_len, sentiment_dim)
            
        Returns:
            Fused features (batch_size, seq_len, fusion_dim)
        """
        # Project to fusion dimension
        market_proj = self.market_proj(market_features)
        sentiment_proj = self.sentiment_proj(sentiment_features)
        
        # Cross-attention
        market_attended, _ = self.market_to_sentiment(
            market_proj, sentiment_proj, sentiment_proj
        )
        sentiment_attended, _ = self.sentiment_to_market(
            sentiment_proj, market_proj, market_proj
        )
        
        # Concatenate and fuse
        fused = torch.cat([market_attended, sentiment_attended], dim=-1)
        fused = self.fusion_ff(fused)
        fused = self.fusion_norm(fused + market_proj)  # Residual connection
        
        return fused


class DuelingHead(nn.Module):
    """
    Dueling network head that separates value and advantage estimation.
    """
    
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        
        self.num_actions = num_actions
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Q-values (batch_size, num_actions)
        """
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values


class MHADQNNetwork(nn.Module):
    """
    Multi-Head Attention Deep Q-Network for Portfolio Optimization.
    
    Architecture:
    1. Input embedding and positional encoding
    2. Multi-head attention transformer blocks
    3. Sentiment fusion module
    4. Dueling network head for Q-value estimation
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract configuration
        self.config = config
        self.market_features = config.get('market_features', 50)
        self.sentiment_features = config.get('sentiment_features', 10)
        self.num_stocks = config.get('num_stocks', 30)
        self.seq_len = config.get('seq_len', 60)
        
        # Model dimensions
        self.d_model = config['attention']['embed_dim']
        self.num_heads = config['attention']['num_heads']
        self.num_layers = config['attention']['num_layers']
        self.dropout = config['attention']['dropout']
        
        # Input embedding
        total_market_features = self.market_features * self.num_stocks
        self.market_embedding = nn.Linear(total_market_features, self.d_model)
        self.sentiment_embedding = nn.Linear(self.sentiment_features * self.num_stocks, 
                                           self.d_model // 2)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=self.seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_model * 4,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Sentiment fusion
        self.sentiment_fusion = SentimentFusionModule(
            market_dim=self.d_model,
            sentiment_dim=self.d_model // 2,
            fusion_dim=self.d_model,
            num_heads=self.num_heads // 2,
            dropout=self.dropout
        )
        
        # Output layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Dueling head for portfolio weights (continuous actions)
        self.dueling_head = DuelingHead(
            input_dim=self.d_model,
            num_actions=self.num_stocks,
            hidden_dim=config['network']['hidden_dims'][0]
        )
        
        # Portfolio constraint layer (softmax for weights)
        self.portfolio_layer = nn.Sequential(
            nn.Linear(self.num_stocks, self.num_stocks),
            nn.Softmax(dim=-1)
        )
        
        self._initialize_weights()
        
        logger.info(f"MHA-DQN initialized with {self.count_parameters():,} parameters")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, market_data: torch.Tensor, sentiment_data: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the MHA-DQN.
        
        Args:
            market_data: Market features (batch_size, seq_len, market_features * num_stocks)
            sentiment_data: Sentiment features (batch_size, seq_len, sentiment_features * num_stocks)
            mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Dictionary containing:
            - q_values: Q-values for portfolio actions
            - portfolio_weights: Normalized portfolio weights
            - attention_weights: Attention weights for interpretability
        """
        batch_size, seq_len = market_data.shape[:2]
        
        # Input embeddings
        market_embedded = self.market_embedding(market_data)  # (batch, seq_len, d_model)
        sentiment_embedded = self.sentiment_embedding(sentiment_data)  # (batch, seq_len, d_model//2)
        
        # Add positional encoding
        market_embedded = market_embedded.transpose(0, 1)  # (seq_len, batch, d_model)
        market_embedded = self.pos_encoding(market_embedded)
        market_embedded = market_embedded.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Apply transformer blocks
        x = market_embedded
        attention_weights = []
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Sentiment fusion
        fused_features = self.sentiment_fusion(x, sentiment_embedded)
        
        # Global pooling (take last timestep or mean)
        if self.config.get('pooling', 'last') == 'mean':
            pooled_features = fused_features.mean(dim=1)
        else:
            pooled_features = fused_features[:, -1, :]  # Last timestep
        
        # Apply layer norm and dropout
        pooled_features = self.layer_norm(pooled_features)
        pooled_features = self.dropout_layer(pooled_features)
        
        # Dueling head for Q-values
        q_values = self.dueling_head(pooled_features)
        
        # Portfolio weights (normalized)
        portfolio_weights = self.portfolio_layer(q_values)
        
        return {
            'q_values': q_values,
            'portfolio_weights': portfolio_weights,
            'attention_weights': attention_weights,
            'features': pooled_features
        }
    
    def get_portfolio_action(self, market_data: torch.Tensor, 
                           sentiment_data: torch.Tensor,
                           epsilon: float = 0.0) -> torch.Tensor:
        """
        Get portfolio action (weights) with optional epsilon-greedy exploration.
        
        Args:
            market_data: Market features
            sentiment_data: Sentiment features
            epsilon: Exploration probability
            
        Returns:
            Portfolio weights tensor
        """
        with torch.no_grad():
            output = self.forward(market_data, sentiment_data)
            portfolio_weights = output['portfolio_weights']
            
            if epsilon > 0 and torch.rand(1).item() < epsilon:
                # Random exploration: uniform random weights
                random_weights = torch.rand_like(portfolio_weights)
                portfolio_weights = F.softmax(random_weights, dim=-1)
            
            return portfolio_weights


class MHADQNAgent:
    """
    Complete MHA-DQN agent with training and inference capabilities.
    """
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        self.config = config
        self.device = device
        
        # Networks
        self.q_network = MHADQNNetwork(config).to(device)
        self.target_network = MHADQNNetwork(config).to(device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['optimizer']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['scheduler']['T_max'],
            eta_min=config['training']['scheduler']['eta_min']
        )
        
        # Training parameters
        self.epsilon = config['training']['exploration']['epsilon_start']
        self.epsilon_decay = config['training']['exploration']['epsilon_decay']
        self.epsilon_min = config['training']['exploration']['epsilon_end']
        
        self.update_target_frequency = config['training']['target_network']['update_frequency']
        self.tau = config['training']['target_network']['tau']
        self.soft_update = config['training']['target_network']['soft_update']
        
        self.step_count = 0
        
        logger.info("MHA-DQN Agent initialized")
    
    def select_action(self, market_data: torch.Tensor, sentiment_data: torch.Tensor,
                     training: bool = True) -> torch.Tensor:
        """Select portfolio action using epsilon-greedy policy."""
        epsilon = self.epsilon if training else 0.0
        return self.q_network.get_portfolio_action(market_data, sentiment_data, epsilon)
    
    def update_target_network(self):
        """Update target network weights."""
        if self.soft_update:
            # Soft update
            for target_param, param in zip(self.target_network.parameters(), 
                                         self.q_network.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
        else:
            # Hard update
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        self.q_network.train()
        
        # Extract batch data
        market_data = batch['market_data'].to(self.device)
        sentiment_data = batch['sentiment_data'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_market_data = batch['next_market_data'].to(self.device)
        next_sentiment_data = batch['next_sentiment_data'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Current Q-values
        current_output = self.q_network(market_data, sentiment_data)
        current_q_values = current_output['q_values']
        
        # Next Q-values from target network
        with torch.no_grad():
            next_output = self.target_network(next_market_data, next_sentiment_data)
            next_q_values = next_output['q_values']
            next_actions = next_output['portfolio_weights']
            
            # Compute target Q-values
            target_q_values = rewards + (1 - dones) * self.config['training'].get('gamma', 0.99) * \
                            torch.sum(next_q_values * next_actions, dim=-1, keepdim=True)
        
        # Compute current Q-values for taken actions
        current_q_values_for_actions = torch.sum(current_q_values * actions, dim=-1, keepdim=True)
        
        # Compute loss
        loss = F.mse_loss(current_q_values_for_actions, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if 'gradient_clipping' in self.config['training']['regularization']:
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(),
                self.config['training']['regularization']['gradient_clipping']
            )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_frequency == 0:
            self.update_target_network()
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'q_values_mean': current_q_values.mean().item(),
            'target_q_values_mean': target_q_values.mean().item()
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        
        logger.info(f"Checkpoint loaded from {filepath}")


def main():
    """Test the MHA-DQN implementation."""
    # Test configuration
    config = {
        'market_features': 50,
        'sentiment_features': 10,
        'num_stocks': 30,
        'seq_len': 60,
        'attention': {
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1
        },
        'network': {
            'hidden_dims': [256, 128]
        },
        'training': {
            'learning_rate': 0.0001,
            'optimizer': {'weight_decay': 0.0001},
            'scheduler': {'T_max': 10000, 'eta_min': 0.00001},
            'exploration': {'epsilon_start': 1.0, 'epsilon_decay': 0.995, 'epsilon_end': 0.01},
            'target_network': {'update_frequency': 100, 'tau': 0.005, 'soft_update': True},
            'regularization': {'gradient_clipping': 1.0}
        }
    }
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = MHADQNAgent(config, device)
    
    # Test forward pass
    batch_size = 32
    seq_len = 60
    market_data = torch.randn(batch_size, seq_len, 50 * 30)
    sentiment_data = torch.randn(batch_size, seq_len, 10 * 30)
    
    # Get portfolio action
    portfolio_weights = agent.select_action(market_data, sentiment_data)
    print(f"Portfolio weights shape: {portfolio_weights.shape}")
    print(f"Portfolio weights sum: {portfolio_weights.sum(dim=-1)}")
    
    print("MHA-DQN test completed successfully!")


if __name__ == "__main__":
    main()
