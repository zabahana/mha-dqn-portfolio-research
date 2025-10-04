# MHA-DQN Portfolio Optimization

A Multi-Head Attention Deep Q-Network implementation for portfolio optimization using reinforcement learning.

## Overview

This project implements a sophisticated portfolio optimization system using Multi-Head Attention Deep Q-Networks (MHA-DQN). The system leverages transformer-inspired attention mechanisms to capture temporal dependencies in financial time series data and make intelligent portfolio allocation decisions.

## Key Features

- **Multi-Head Attention DQN**: Advanced neural network architecture with attention mechanisms
- **Portfolio Environment**: Custom reinforcement learning environment for portfolio management
- **Feature Engineering**: Comprehensive technical and fundamental analysis features
- **Experience Replay**: Prioritized experience replay buffer for stable training
- **Risk Management**: Built-in risk-aware reward functions and portfolio constraints

## Dataset

### Stock Universe (9 Stocks)
- **Large Cap Stocks**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, UNH, JNJ
- **Data Period**: 5 years (2020-2024)
- **Features**: 30 features per stock including technical indicators and fundamental data

### Data Sources
- **Price Data**: Alpha Vantage API
- **Fundamental Data**: Alpha Vantage Fundamental Data API
- **Technical Indicators**: Computed from price data

## Architecture

### MHA-DQN Components

1. **Multi-Head Attention Module**
   - Self-attention for temporal feature relationships
   - Cross-attention for feature interactions
   - Positional encoding for sequence understanding

2. **Portfolio Optimization Module**
   - Dueling network architecture
   - Risk-aware reward function
   - Portfolio weight constraints

3. **Experience Replay**
   - Prioritized replay buffer
   - Batch sampling for stable training

## Installation & Setup

### Prerequisites
```bash
Python 3.9+
PyTorch 2.0+
Alpha Vantage API Key
```

### Installation
```bash
git clone https://github.com/zabahana/mha-dqn-portfolio-research.git
cd mha-dqn-portfolio-research
pip install -r requirements.txt
```

### Configuration
```bash
# Set your Alpha Vantage API key
export ALPHA_VANTAGE_API_KEY="your_api_key_here"

# Configure data paths
export DATA_PATH="./data"
export RESULTS_PATH="./results"
```

## Usage

### 1. Data Collection
```bash
python scripts/collect_data.py
```

### 2. Feature Engineering
```bash
python scripts/feature_engineering.py
```

### 3. Model Training
```bash
python scripts/train_mha_dqn.py --config configs/mha_dqn_config.yaml --episodes 100
```

### 4. Model Evaluation
```bash
python scripts/evaluate_model.py --model-path models/final_model.pth
```

## Project Structure

```
mha-dqn-portfolio-research/
├── README.md
├── requirements.txt
├── configs/
│   └── mha_dqn_config.yaml
├── src/
│   ├── data/
│   │   └── alpha_vantage_client.py
│   ├── models/
│   │   ├── mha_dqn.py
│   │   └── attention_modules.py
│   ├── training/
│   │   ├── environment.py
│   │   └── replay_buffer.py
│   └── utils/
│       └── logging.py
├── scripts/
│   ├── collect_data.py
│   ├── feature_engineering.py
│   └── train_mha_dqn.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/
│   └── checkpoints/
└── results/
    └── training_metrics.json
```

## Model Performance

The trained MHA-DQN model achieves:
- **Best Sharpe Ratio**: 1.60
- **Average Sharpe Ratio**: 0.44
- **Model Parameters**: 23M+ parameters
- **Training Episodes**: 100 episodes with 252 steps each

## Configuration

The model configuration is defined in `configs/mha_dqn_config.yaml`:

- **Model Architecture**: Attention heads, hidden dimensions, network depth
- **Training Parameters**: Learning rate, batch size, replay buffer size
- **Environment Settings**: Portfolio constraints, reward weights
- **Data Settings**: Time periods, feature selection

## Results

Training results are saved in the `results/` directory:
- **Training Metrics**: `training_metrics.json` with detailed performance logs
- **Model Checkpoints**: Best performing models in `models/`
- **Visualizations**: Training progress plots and performance charts

## License

This project is licensed under the MIT License.

## Contact

For questions about this implementation, please contact:
- **Email**: zga5029@psu.edu
- **Institution**: Penn State University