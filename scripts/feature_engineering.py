#!/usr/bin/env python3
"""
Feature Engineering Script for MHA-DQN Portfolio Optimization Research
Processes raw data and creates features for model training

Usage:
    python scripts/feature_engineering.py --include-sentiment --technical-indicators
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import load_config
from utils.logging import setup_logging

# Technical analysis library
import ta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for MHA-DQN portfolio optimization.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Data directories
        self.data_dir = Path(config.get('data_dir', './data'))
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.features_dir = self.data_dir / 'features'
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Stock universe
        self.stock_universe = {
            'large_cap': config['data']['stocks']['large_cap'],
            'mid_cap': config['data']['stocks']['mid_cap'],
            'small_cap': config['data']['stocks']['small_cap']
        }
        
        logger.info("FeatureEngineer initialized")
    
    def load_price_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Load price data for given symbols."""
        logger.info(f"Loading price data for {len(symbols)} symbols")
        
        price_data = {}
        for symbol in symbols:
            price_file = self.raw_dir / f"{symbol}_prices.csv"
            
            if price_file.exists():
                try:
                    df = pd.read_csv(price_file, index_col=0, parse_dates=True)
                    df = df.sort_index()
                    price_data[symbol] = df
                    logger.info(f"Loaded {len(df)} days of price data for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading price data for {symbol}: {e}")
            else:
                logger.warning(f"Price data file not found for {symbol}")
        
        return price_data
    
    def load_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Load fundamental data for given symbols."""
        logger.info(f"Loading fundamental data for {len(symbols)} symbols")
        
        fundamental_data = {}
        for symbol in symbols:
            fund_file = self.raw_dir / f"{symbol}_fundamentals.json"
            
            if fund_file.exists():
                try:
                    with open(fund_file, 'r') as f:
                        data = json.load(f)
                    fundamental_data[symbol] = data
                    logger.info(f"Loaded fundamental data for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading fundamental data for {symbol}: {e}")
            else:
                logger.warning(f"Fundamental data file not found for {symbol}")
        
        return fundamental_data
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for a price DataFrame."""
        try:
            # Ensure we have the required columns
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                logger.warning("Missing required columns for technical indicators")
                return df
            
            # Create a copy to avoid modifying original
            result = df.copy()
            
            # Price-based indicators
            result['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            result['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            result['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            result['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # Momentum indicators
            result['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
            result['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            result['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
            
            # MACD
            macd_line, macd_signal, macd_histogram = ta.trend.MACD(df['close']).macd(), ta.trend.MACD(df['close']).macd_signal(), ta.trend.MACD(df['close']).macd_diff()
            result['macd'] = macd_line
            result['macd_signal'] = macd_signal
            result['macd_histogram'] = macd_histogram
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            result['bb_upper'] = bb.bollinger_hband()
            result['bb_middle'] = bb.bollinger_mavg()
            result['bb_lower'] = bb.bollinger_lband()
            result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
            result['bb_position'] = (df['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
            
            # Volatility indicators
            result['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            
            # Volume indicators
            result['volume_sma_20'] = ta.volume.volume_sma(df['close'], df['volume'], window=20)
            result['volume_ratio'] = df['volume'] / result['volume_sma_20']
            
            # Price action features
            result['returns'] = df['close'].pct_change()
            result['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            result['volatility_20'] = result['returns'].rolling(window=20).std()
            result['volatility_60'] = result['returns'].rolling(window=60).std()
            
            # Price patterns
            result['high_low_ratio'] = df['high'] / df['low']
            result['close_open_ratio'] = df['close'] / df['open']
            result['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Trend indicators
            result['price_above_sma20'] = (df['close'] > result['sma_20']).astype(int)
            result['price_above_sma50'] = (df['close'] > result['sma_50']).astype(int)
            result['sma20_above_sma50'] = (result['sma_20'] > result['sma_50']).astype(int)
            
            logger.info(f"Calculated {len([col for col in result.columns if col not in df.columns])} technical indicators")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def extract_fundamental_features(self, fundamental_data: Dict[str, Dict]) -> pd.DataFrame:
        """Extract fundamental features from fundamental data."""
        logger.info("Extracting fundamental features")
        
        features = []
        
        for symbol, data in fundamental_data.items():
            try:
                feature_dict = {'symbol': symbol}
                
                # Financial ratios
                feature_dict['pe_ratio'] = self._safe_float(data.get('PERatio', 0))
                feature_dict['pb_ratio'] = self._safe_float(data.get('PriceToBookRatio', 0))
                feature_dict['ps_ratio'] = self._safe_float(data.get('PriceToSalesRatioTTM', 0))
                feature_dict['peg_ratio'] = self._safe_float(data.get('PEGRatio', 0))
                
                # Profitability metrics
                feature_dict['profit_margin'] = self._safe_float(data.get('ProfitMargin', 0))
                feature_dict['operating_margin'] = self._safe_float(data.get('OperatingMarginTTM', 0))
                feature_dict['roe'] = self._safe_float(data.get('ReturnOnEquityTTM', 0))
                feature_dict['roa'] = self._safe_float(data.get('ReturnOnAssetsTTM', 0))
                
                # Growth metrics
                feature_dict['revenue_growth'] = self._safe_float(data.get('QuarterlyRevenueGrowthYOY', 0))
                feature_dict['earnings_growth'] = self._safe_float(data.get('QuarterlyEarningsGrowthYOY', 0))
                
                # Financial health
                feature_dict['debt_to_equity'] = self._safe_float(data.get('DebtToEquity', 0))
                feature_dict['current_ratio'] = self._safe_float(data.get('CurrentRatio', 0))
                feature_dict['quick_ratio'] = self._safe_float(data.get('QuickRatio', 0))
                
                # Market metrics
                feature_dict['market_cap'] = self._safe_float(data.get('MarketCapitalization', 0))
                feature_dict['enterprise_value'] = self._safe_float(data.get('EnterpriseValue', 0))
                feature_dict['ev_revenue'] = self._safe_float(data.get('EVToRevenue', 0))
                feature_dict['ev_ebitda'] = self._safe_float(data.get('EVToEBITDA', 0))
                
                # Dividend metrics
                feature_dict['dividend_yield'] = self._safe_float(data.get('DividendYield', 0))
                feature_dict['dividend_per_share'] = self._safe_float(data.get('DividendPerShare', 0))
                feature_dict['payout_ratio'] = self._safe_float(data.get('PayoutRatio', 0))
                
                # Beta and volatility
                feature_dict['beta'] = self._safe_float(data.get('Beta', 1.0))
                feature_dict['52_week_high'] = self._safe_float(data.get('52WeekHigh', 0))
                feature_dict['52_week_low'] = self._safe_float(data.get('52WeekLow', 0))
                
                # Analyst metrics
                feature_dict['analyst_target_price'] = self._safe_float(data.get('AnalystTargetPrice', 0))
                
                features.append(feature_dict)
                
            except Exception as e:
                logger.error(f"Error extracting fundamental features for {symbol}: {e}")
                continue
        
        df = pd.DataFrame(features)
        df = df.set_index('symbol')
        
        logger.info(f"Extracted fundamental features for {len(df)} symbols")
        return df
    
    def _safe_float(self, value, default=0.0) -> float:
        """Safely convert value to float."""
        try:
            if value is None or value == 'None' or value == '':
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def create_market_features(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create market-wide features."""
        logger.info("Creating market features")
        
        # Calculate market indices
        all_returns = []
        market_caps = []
        
        for symbol, df in price_data.items():
            if 'returns' in df.columns:
                returns = df['returns'].dropna()
                returns.name = symbol
                all_returns.append(returns)
        
        if not all_returns:
            logger.warning("No returns data available for market features")
            return pd.DataFrame()
        
        # Combine all returns
        returns_df = pd.concat(all_returns, axis=1)
        returns_df = returns_df.dropna()
        
        # Market features
        market_features = pd.DataFrame(index=returns_df.index)
        
        # Equal-weighted market return
        market_features['market_return_ew'] = returns_df.mean(axis=1)
        
        # Market volatility
        market_features['market_volatility'] = returns_df.std(axis=1)
        
        # Market correlation (average pairwise correlation)
        rolling_corr = returns_df.rolling(window=60).corr()
        market_features['market_correlation'] = rolling_corr.groupby(level=0).mean().mean(axis=1)
        
        # VIX-like measure (volatility of volatility)
        market_vol_20 = returns_df.rolling(window=20).std().mean(axis=1)
        market_features['vix_proxy'] = market_vol_20.rolling(window=20).std()
        
        # Market momentum
        market_features['market_momentum_5d'] = market_features['market_return_ew'].rolling(window=5).sum()
        market_features['market_momentum_20d'] = market_features['market_return_ew'].rolling(window=20).sum()
        
        # Dispersion (cross-sectional volatility)
        market_features['return_dispersion'] = returns_df.std(axis=1)
        
        logger.info(f"Created {len(market_features.columns)} market features")
        return market_features
    
    def create_sentiment_features(self, symbols: List[str]) -> pd.DataFrame:
        """Create sentiment features (placeholder for now)."""
        logger.info("Creating sentiment features (placeholder)")
        
        # For now, create dummy sentiment features
        # In a real implementation, this would process news sentiment data
        
        dates = pd.date_range(start='2023-01-01', end='2025-10-01', freq='D')
        
        sentiment_features = pd.DataFrame(index=dates)
        
        for symbol in symbols:
            # Dummy sentiment scores (normally distributed around 0)
            np.random.seed(hash(symbol) % 2**32)  # Consistent random seed per symbol
            sentiment_features[f'{symbol}_sentiment'] = np.random.normal(0, 0.1, len(dates))
            sentiment_features[f'{symbol}_sentiment_ma5'] = sentiment_features[f'{symbol}_sentiment'].rolling(5).mean()
            sentiment_features[f'{symbol}_sentiment_ma20'] = sentiment_features[f'{symbol}_sentiment'].rolling(20).mean()
        
        # Market-wide sentiment
        sentiment_features['market_sentiment'] = sentiment_features[[col for col in sentiment_features.columns if col.endswith('_sentiment')]].mean(axis=1)
        
        logger.info(f"Created sentiment features for {len(symbols)} symbols")
        return sentiment_features
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """Normalize features using specified method."""
        logger.info(f"Normalizing features using {method}")
        
        result = df.copy()
        
        if method == 'zscore':
            # Z-score normalization
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result[numeric_cols] = (result[numeric_cols] - result[numeric_cols].mean()) / result[numeric_cols].std()
        
        elif method == 'minmax':
            # Min-max normalization
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result[numeric_cols] = (result[numeric_cols] - result[numeric_cols].min()) / (result[numeric_cols].max() - result[numeric_cols].min())
        
        elif method == 'robust':
            # Robust normalization (using median and IQR)
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            median = result[numeric_cols].median()
            q75 = result[numeric_cols].quantile(0.75)
            q25 = result[numeric_cols].quantile(0.25)
            iqr = q75 - q25
            result[numeric_cols] = (result[numeric_cols] - median) / iqr
        
        # Fill any remaining NaN values
        result = result.fillna(0)
        
        logger.info("Feature normalization completed")
        return result
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling."""
        logger.info(f"Creating sequences with length {sequence_length}")
        
        # Sort by date
        df = df.sort_index()
        
        # Convert to numpy array
        data = df.values
        
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(data)):
            # Input sequence
            seq = data[i-sequence_length:i]
            sequences.append(seq)
            
            # Target (next day's features - for now just use the same features)
            target = data[i]
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"Created {len(sequences)} sequences")
        return sequences, targets
    
    def run_feature_engineering(self, symbols: List[str], include_sentiment: bool = True, 
                               include_technical: bool = True) -> Dict[str, pd.DataFrame]:
        """Run complete feature engineering pipeline."""
        logger.info(f"Starting feature engineering for {len(symbols)} symbols")
        
        results = {}
        
        # 1. Load raw data
        logger.info("=== Loading Raw Data ===")
        price_data = self.load_price_data(symbols)
        fundamental_data = self.load_fundamental_data(symbols)
        
        # 2. Calculate technical indicators
        if include_technical:
            logger.info("=== Calculating Technical Indicators ===")
            for symbol in price_data:
                price_data[symbol] = self.calculate_technical_indicators(price_data[symbol])
        
        # 3. Extract fundamental features
        logger.info("=== Extracting Fundamental Features ===")
        fundamental_features = self.extract_fundamental_features(fundamental_data)
        results['fundamental_features'] = fundamental_features
        
        # 4. Create market features
        logger.info("=== Creating Market Features ===")
        market_features = self.create_market_features(price_data)
        results['market_features'] = market_features
        
        # 5. Create sentiment features
        if include_sentiment:
            logger.info("=== Creating Sentiment Features ===")
            sentiment_features = self.create_sentiment_features(symbols)
            results['sentiment_features'] = sentiment_features
        
        # 6. Combine all features for each symbol
        logger.info("=== Combining Features ===")
        combined_features = {}
        
        for symbol in symbols:
            if symbol not in price_data:
                continue
                
            # Start with price and technical data
            symbol_features = price_data[symbol].copy()
            
            # Add market features
            if not market_features.empty:
                symbol_features = symbol_features.join(market_features, how='left')
            
            # Add sentiment features
            if include_sentiment and not sentiment_features.empty:
                sentiment_cols = [col for col in sentiment_features.columns if col.startswith(symbol) or col == 'market_sentiment']
                symbol_features = symbol_features.join(sentiment_features[sentiment_cols], how='left')
            
            # Add fundamental features (broadcast to all dates)
            if symbol in fundamental_features.index:
                fund_data = fundamental_features.loc[symbol]
                for col, value in fund_data.items():
                    symbol_features[f'fund_{col}'] = value
            
            # Fill NaN values
            symbol_features = symbol_features.fillna(method='ffill').fillna(0)
            
            combined_features[symbol] = symbol_features
        
        results['combined_features'] = combined_features
        
        # 7. Save processed data
        logger.info("=== Saving Processed Data ===")
        for symbol, df in combined_features.items():
            filepath = self.processed_dir / f"{symbol}_features.csv"
            df.to_csv(filepath)
            logger.info(f"Saved features for {symbol}: {df.shape}")
        
        # Save fundamental features
        fundamental_features.to_csv(self.processed_dir / "fundamental_features.csv")
        
        # Save market features
        if not market_features.empty:
            market_features.to_csv(self.processed_dir / "market_features.csv")
        
        # Save sentiment features
        if include_sentiment and not sentiment_features.empty:
            sentiment_features.to_csv(self.processed_dir / "sentiment_features.csv")
        
        # 8. Create normalized features
        logger.info("=== Creating Normalized Features ===")
        normalized_features = {}
        
        for symbol, df in combined_features.items():
            normalized_df = self.normalize_features(df, method='robust')
            normalized_features[symbol] = normalized_df
            
            # Save normalized features
            filepath = self.features_dir / f"{symbol}_normalized.csv"
            normalized_df.to_csv(filepath)
        
        results['normalized_features'] = normalized_features
        
        logger.info("Feature engineering completed!")
        return results


def main():
    """Main function for feature engineering script."""
    parser = argparse.ArgumentParser(description='Feature engineering for MHA-DQN research')
    parser.add_argument('--config', type=str, default='configs/mha_dqn_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--include-sentiment', action='store_true',
                       help='Include sentiment features')
    parser.add_argument('--include-technical', action='store_true', default=True,
                       help='Include technical indicators')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to process (default: all available)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize feature engineer
    engineer = FeatureEngineer(config)
    
    # Get symbols to process
    if args.symbols:
        symbols = args.symbols
    else:
        # Get all symbols with available data
        symbols = []
        for category in engineer.stock_universe.values():
            symbols.extend(category)
        
        # Filter to only symbols with available data
        available_symbols = []
        for symbol in symbols:
            price_file = engineer.raw_dir / f"{symbol}_prices.csv"
            if price_file.exists():
                available_symbols.append(symbol)
        
        symbols = available_symbols
    
    logger.info(f"Processing features for {len(symbols)} symbols: {symbols}")
    
    # Run feature engineering
    try:
        results = engineer.run_feature_engineering(
            symbols=symbols,
            include_sentiment=args.include_sentiment,
            include_technical=args.include_technical
        )
        
        print("\n" + "="*50)
        print("FEATURE ENGINEERING COMPLETED")
        print("="*50)
        print(f"Symbols processed: {len(symbols)}")
        print(f"Combined features: {len(results.get('combined_features', {}))}")
        print(f"Normalized features: {len(results.get('normalized_features', {}))}")
        
        # Print feature summary
        if 'combined_features' in results:
            sample_symbol = list(results['combined_features'].keys())[0]
            sample_df = results['combined_features'][sample_symbol]
            print(f"Features per symbol: {sample_df.shape[1]}")
            print(f"Time periods: {sample_df.shape[0]}")
            print(f"Date range: {sample_df.index.min()} to {sample_df.index.max()}")
        
        print(f"\nProcessed data saved to: {engineer.processed_dir}")
        print(f"Normalized features saved to: {engineer.features_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
