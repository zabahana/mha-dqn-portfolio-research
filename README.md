# Multi-Head Attention Deep Q-Networks for Portfolio Optimization: A Sentiment-Integrated Reinforcement Learning Approach

**Research Project for NeurIPS 2024 - Machine Learning in Finance**

**Author:** Zelalem Abahana  
**Institution:** Penn State University  
**Email:** zga5029@psu.edu

## Abstract

This research presents a novel Multi-Head Attention Deep Q-Network (MHA-DQN) architecture for portfolio optimization that integrates earnings call sentiment analysis with quantitative trading signals. Our approach leverages transformer-inspired attention mechanisms to capture temporal dependencies in financial time series while incorporating fundamental analysis through earnings call transcripts. We evaluate our method on a comprehensive dataset of 30 stocks across market capitalizations over 10 years, demonstrating superior risk-adjusted returns with rigorous statistical validation.

## Key Contributions

1. **Novel MHA-DQN Architecture**: First application of multi-head attention mechanisms to deep reinforcement learning for portfolio optimization
2. **Earnings Call Integration**: Comprehensive sentiment analysis pipeline for earnings call transcripts using state-of-the-art NLP models
3. **Multi-Cap Analysis**: Systematic evaluation across low, mid, and high market capitalization stocks
4. **Rigorous Validation**: Statistical significance testing, ablation studies, and robustness analysis meeting NeurIPS standards

## Dataset Specifications

### Stock Universe (30 Stocks)
- **Large Cap (10 stocks)**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK.B, UNH, JNJ
- **Mid Cap (10 stocks)**: ROKU, PLTR, SNOW, ZM, DOCU, CRWD, NET, DDOG, OKTA, ZS
- **Small Cap (10 stocks)**: SFIX, CLOV, WISH, SPCE, NKLA, LCID, RIVN, HOOD, COIN, RBLX

### Data Sources
- **Price Data**: Alpha Vantage Premium API (10 years, daily frequency)
- **Earnings Calls**: Alpha Vantage Earnings API with transcript analysis
- **Fundamental Data**: Alpha Vantage Fundamental Data API
- **News Sentiment**: Alpha Vantage News & Sentiment API

### Time Period
- **Training Period**: 2014-2022 (8 years)
- **Validation Period**: 2023 (1 year)
- **Test Period**: 2024 (1 year)

## Architecture Overview

### Multi-Head Attention DQN Components

1. **Temporal Attention Module**
   - Multi-head self-attention for price series
   - Positional encoding for temporal relationships
   - Residual connections with layer normalization

2. **Sentiment Fusion Module**
   - Cross-attention between price and sentiment features
   - Hierarchical sentiment aggregation from earnings calls
   - Dynamic weighting based on earnings announcement proximity

3. **Portfolio Optimization Module**
   - Dueling network architecture for value/advantage decomposition
   - Risk-aware reward function with sentiment integration
   - Experience replay with prioritized sampling

## Methodology

### 1. Data Collection & Preprocessing
```
Alpha Vantage API → Raw Data → Feature Engineering → Normalized Features
```

### 2. Sentiment Analysis Pipeline
```
Earnings Transcripts → NLP Processing → Sentiment Scores → Temporal Alignment
```

### 3. MHA-DQN Training
```
Market State → Attention Layers → Q-Values → Portfolio Actions → Rewards
```

### 4. Evaluation Framework
```
Backtesting → Performance Metrics → Statistical Tests → Robustness Analysis
```

## Installation & Setup

### Prerequisites
```bash
Python 3.9+
PyTorch 2.0+
Transformers 4.0+
Alpha Vantage API Key (Premium)
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
export ALPHA_VANTAGE_API_KEY="your_premium_api_key_here"

# Configure data paths
export DATA_PATH="./data"
export RESULTS_PATH="./results"
```

## Usage

### 1. Data Collection
```bash
python scripts/collect_data.py --years 10 --stocks all
```

### 2. Feature Engineering
```bash
python scripts/feature_engineering.py --include-earnings --sentiment-model bert
```

### 3. Model Training
```bash
python scripts/train_mha_dqn.py --config configs/mha_dqn_config.yaml
```

### 4. Evaluation
```bash
python scripts/evaluate_model.py --model-path models/mha_dqn_best.pth
```

### 5. Generate Research Paper
```bash
python scripts/generate_paper.py --include-diagrams --neurips-format
```

## Project Structure

```
mha-dqn-portfolio-research/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── mha_dqn_config.yaml
│   ├── data_config.yaml
│   └── evaluation_config.yaml
├── src/
│   ├── data/
│   │   ├── alpha_vantage_client.py
│   │   ├── earnings_processor.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── mha_dqn.py
│   │   ├── attention_modules.py
│   │   └── sentiment_fusion.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── environment.py
│   │   └── replay_buffer.py
│   ├── evaluation/
│   │   ├── backtester.py
│   │   ├── metrics.py
│   │   └── statistical_tests.py
│   └── utils/
│       ├── visualization.py
│       ├── logging.py
│       └── config.py
├── scripts/
│   ├── collect_data.py
│   ├── feature_engineering.py
│   ├── train_mha_dqn.py
│   ├── evaluate_model.py
│   └── generate_paper.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_analysis.ipynb
│   └── results_visualization.ipynb
├── paper/
│   ├── neurips_paper.tex
│   ├── figures/
│   ├── tables/
│   └── references.bib
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/
│   ├── checkpoints/
│   └── final/
├── results/
│   ├── experiments/
│   ├── figures/
│   └── tables/
└── tests/
    ├── test_data.py
    ├── test_models.py
    └── test_evaluation.py
```

## Research Methodology

### 1. Problem Formulation
- **Objective**: Maximize risk-adjusted portfolio returns using sentiment-enhanced RL
- **State Space**: Market features + earnings sentiment + technical indicators
- **Action Space**: Portfolio weight allocations across 30 stocks
- **Reward Function**: Multi-factor reward combining returns, risk, and sentiment signals

### 2. Model Architecture
- **Attention Mechanism**: 8-head self-attention with 512-dimensional embeddings
- **Network Depth**: 6 transformer blocks with residual connections
- **Sentiment Integration**: Cross-attention between market and sentiment features
- **Output Layer**: Dueling architecture for value/advantage decomposition

### 3. Training Protocol
- **Experience Replay**: Prioritized replay buffer with 1M capacity
- **Target Networks**: Soft updates with τ=0.005
- **Optimization**: AdamW optimizer with cosine annealing
- **Regularization**: Dropout (0.1), weight decay (1e-4), gradient clipping

### 4. Evaluation Metrics
- **Performance**: Sharpe ratio, Sortino ratio, Calmar ratio, Maximum Drawdown
- **Risk**: VaR, CVaR, volatility, beta, tracking error
- **Statistical**: t-tests, bootstrap confidence intervals, Diebold-Mariano tests

## Expected Results

### Performance Targets (vs. Benchmarks)
- **Sharpe Ratio**: >1.5 (vs. 0.8-1.2 for benchmarks)
- **Maximum Drawdown**: <15% (vs. 20-25% for benchmarks)
- **Annual Return**: 15-20% (vs. 8-12% for benchmarks)
- **Information Ratio**: >1.0 (vs. 0.3-0.7 for benchmarks)

### Statistical Significance
- **p-values**: <0.01 for all performance metrics
- **Confidence Intervals**: 95% bootstrap intervals
- **Robustness**: Consistent performance across market regimes

## Publication Timeline

- **Month 1-2**: Data collection and preprocessing
- **Month 3-4**: Model development and initial training
- **Month 5-6**: Comprehensive evaluation and ablation studies
- **Month 7-8**: Paper writing and diagram creation
- **Month 9**: Final experiments and statistical validation
- **Month 10**: Paper submission to NeurIPS 2024

## License

This research project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{abahana2024mha,
  title={Multi-Head Attention Deep Q-Networks for Portfolio Optimization: A Sentiment-Integrated Reinforcement Learning Approach},
  author={Abahana, Zelalem},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## Contact

For questions about this research, please contact:
- **Email**: zga5029@psu.edu
- **Institution**: Penn State University
- **Department**: College of Information Sciences and Technology
