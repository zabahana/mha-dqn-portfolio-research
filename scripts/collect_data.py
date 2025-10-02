#!/usr/bin/env python3
"""
Data Collection Script for MHA-DQN Portfolio Optimization Research
Collects 10 years of data for 30 stocks across market caps using Alpha Vantage API

Usage:
    python scripts/collect_data.py --years 10 --stocks all
    python scripts/collect_data.py --years 5 --stocks large_cap
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.alpha_vantage_client import AlphaVantageClient
from utils.config import load_config
from utils.logging import setup_logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Comprehensive data collector for the MHA-DQN research project.
    """
    
    def __init__(self, config: Dict, api_key: str):
        self.config = config
        self.api_key = api_key
        self.client = AlphaVantageClient(api_key=api_key, premium=True)
        
        # Stock universe
        self.stock_universe = {
            'large_cap': config['data']['stocks']['large_cap'],
            'mid_cap': config['data']['stocks']['mid_cap'],
            'small_cap': config['data']['stocks']['small_cap']
        }
        
        # Data directories
        self.data_dir = Path(config.get('data_dir', './data'))
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DataCollector initialized")
    
    def get_stock_list(self, category: str = 'all') -> List[str]:
        """Get list of stocks to collect data for."""
        if category == 'all':
            stocks = []
            for cat_stocks in self.stock_universe.values():
                stocks.extend(cat_stocks)
            return stocks
        elif category in self.stock_universe:
            return self.stock_universe[category]
        else:
            raise ValueError(f"Unknown stock category: {category}")
    
    def collect_price_data(self, symbols: List[str], years: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Collect historical price data for given symbols.
        
        Args:
            symbols: List of stock symbols
            years: Number of years of historical data
            
        Returns:
            Dictionary mapping symbols to price DataFrames
        """
        logger.info(f"Collecting price data for {len(symbols)} symbols ({years} years)")
        
        price_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Collecting price data for {symbol} ({i+1}/{len(symbols)})")
            
            try:
                # Get daily price data
                data = self.client.get_daily_prices(symbol, outputsize='full')
                
                if not data.empty:
                    # Filter by date range
                    mask = (data.index >= start_date) & (data.index <= end_date)
                    filtered_data = data.loc[mask].copy()
                    
                    if not filtered_data.empty:
                        price_data[symbol] = filtered_data
                        
                        # Save individual file
                        filepath = self.raw_dir / f"{symbol}_prices.csv"
                        filtered_data.to_csv(filepath)
                        
                        logger.info(f"Saved {len(filtered_data)} days of price data for {symbol}")
                    else:
                        logger.warning(f"No data in date range for {symbol}")
                else:
                    logger.warning(f"No price data retrieved for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error collecting price data for {symbol}: {e}")
                continue
            
            # Small delay to be respectful to API
            time.sleep(0.5)
        
        logger.info(f"Price data collection completed: {len(price_data)} symbols")
        return price_data
    
    def collect_earnings_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Collect earnings data and calendar for given symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to earnings data
        """
        logger.info(f"Collecting earnings data for {len(symbols)} symbols")
        
        earnings_data = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Collecting earnings data for {symbol} ({i+1}/{len(symbols)})")
            
            try:
                # Get quarterly and annual earnings
                quarterly, annual = self.client.get_earnings_data(symbol)
                
                symbol_earnings = {
                    'quarterly': quarterly,
                    'annual': annual
                }
                
                earnings_data[symbol] = symbol_earnings
                
                # Save individual files
                if not quarterly.empty:
                    filepath = self.raw_dir / f"{symbol}_earnings_quarterly.csv"
                    quarterly.to_csv(filepath)
                
                if not annual.empty:
                    filepath = self.raw_dir / f"{symbol}_earnings_annual.csv"
                    annual.to_csv(filepath)
                
                logger.info(f"Saved earnings data for {symbol}: {len(quarterly)} quarterly, {len(annual)} annual")
                
            except Exception as e:
                logger.error(f"Error collecting earnings data for {symbol}: {e}")
                continue
            
            time.sleep(0.5)
        
        logger.info(f"Earnings data collection completed: {len(earnings_data)} symbols")
        return earnings_data
    
    def collect_news_sentiment(self, symbols: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Collect news sentiment data for given symbols.
        
        Args:
            symbols: List of stock symbols
            days: Number of days of news history
            
        Returns:
            Dictionary mapping symbols to news sentiment DataFrames
        """
        logger.info(f"Collecting news sentiment for {len(symbols)} symbols ({days} days)")
        
        sentiment_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        time_from = start_date.strftime('%Y%m%dT0000')
        time_to = end_date.strftime('%Y%m%dT2359')
        
        # Collect in batches to optimize API calls
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            logger.info(f"Collecting news sentiment for batch {i//batch_size + 1}: {batch_symbols}")
            
            try:
                # Get news sentiment for batch
                news_data = self.client.get_news_sentiment(
                    tickers=batch_symbols,
                    time_from=time_from,
                    time_to=time_to,
                    limit=1000
                )
                
                if not news_data.empty:
                    # Split by symbol and save
                    for symbol in batch_symbols:
                        # Filter articles mentioning this symbol
                        symbol_cols = [col for col in news_data.columns if col.startswith(f'{symbol}_')]
                        if symbol_cols:
                            symbol_data = news_data[['time_published', 'title', 'summary', 
                                                   'overall_sentiment_score', 'overall_sentiment_label'] + symbol_cols].copy()
                            
                            # Remove rows where this symbol isn't mentioned
                            symbol_data = symbol_data.dropna(subset=symbol_cols, how='all')
                            
                            if not symbol_data.empty:
                                sentiment_data[symbol] = symbol_data
                                
                                # Save individual file
                                filepath = self.raw_dir / f"{symbol}_news_sentiment.csv"
                                symbol_data.to_csv(filepath, index=False)
                                
                                logger.info(f"Saved {len(symbol_data)} news articles for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting news sentiment for batch {batch_symbols}: {e}")
                continue
            
            time.sleep(2)  # Longer delay for news API
        
        logger.info(f"News sentiment collection completed: {len(sentiment_data)} symbols")
        return sentiment_data
    
    def collect_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Collect fundamental data for given symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to fundamental data
        """
        logger.info(f"Collecting fundamental data for {len(symbols)} symbols")
        
        fundamental_data = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Collecting fundamental data for {symbol} ({i+1}/{len(symbols)})")
            
            try:
                # Get company overview
                overview = self.client.get_company_overview(symbol)
                
                if overview:
                    fundamental_data[symbol] = overview
                    
                    # Save individual file
                    filepath = self.raw_dir / f"{symbol}_fundamentals.json"
                    with open(filepath, 'w') as f:
                        json.dump(overview, f, indent=2, default=str)
                    
                    logger.info(f"Saved fundamental data for {symbol}")
                else:
                    logger.warning(f"No fundamental data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting fundamental data for {symbol}: {e}")
                continue
            
            time.sleep(0.5)
        
        logger.info(f"Fundamental data collection completed: {len(fundamental_data)} symbols")
        return fundamental_data
    
    def collect_technical_indicators(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Collect technical indicators for given symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to technical indicator data
        """
        logger.info(f"Collecting technical indicators for {len(symbols)} symbols")
        
        # Technical indicators to collect
        indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS', 'ATR']
        indicator_params = {
            'SMA': {'time_period': 20},
            'EMA': {'time_period': 12},
            'RSI': {'time_period': 14},
            'MACD': {},
            'BBANDS': {'time_period': 20},
            'ATR': {'time_period': 14}
        }
        
        technical_data = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Collecting technical indicators for {symbol} ({i+1}/{len(symbols)})")
            
            symbol_indicators = {}
            
            for indicator in indicators:
                try:
                    params = indicator_params.get(indicator, {})
                    data = self.client.get_technical_indicators(
                        symbol=symbol,
                        indicator=indicator,
                        interval='daily',
                        **params
                    )
                    
                    if not data.empty:
                        symbol_indicators[indicator] = data
                        
                        # Save individual file
                        filepath = self.raw_dir / f"{symbol}_{indicator.lower()}.csv"
                        data.to_csv(filepath)
                        
                        logger.info(f"Saved {indicator} data for {symbol}: {len(data)} points")
                    
                except Exception as e:
                    logger.error(f"Error collecting {indicator} for {symbol}: {e}")
                    continue
                
                time.sleep(0.3)  # Small delay between indicators
            
            if symbol_indicators:
                technical_data[symbol] = symbol_indicators
        
        logger.info(f"Technical indicators collection completed: {len(technical_data)} symbols")
        return technical_data
    
    def create_data_summary(self, symbols: List[str]) -> Dict:
        """
        Create a summary of collected data.
        
        Args:
            symbols: List of symbols data was collected for
            
        Returns:
            Dictionary with data summary
        """
        summary = {
            'collection_date': datetime.now().isoformat(),
            'symbols': symbols,
            'total_symbols': len(symbols),
            'data_types': [],
            'file_counts': {},
            'date_ranges': {}
        }
        
        # Check what data types were collected
        data_types = ['prices', 'earnings_quarterly', 'earnings_annual', 
                     'news_sentiment', 'fundamentals']
        
        for data_type in data_types:
            files = list(self.raw_dir.glob(f"*_{data_type}.*"))
            if files:
                summary['data_types'].append(data_type)
                summary['file_counts'][data_type] = len(files)
        
        # Check date ranges for price data
        price_files = list(self.raw_dir.glob("*_prices.csv"))
        if price_files:
            min_date = None
            max_date = None
            
            for file in price_files[:5]:  # Sample a few files
                try:
                    df = pd.read_csv(file, index_col=0, parse_dates=True)
                    if not df.empty:
                        file_min = df.index.min()
                        file_max = df.index.max()
                        
                        if min_date is None or file_min < min_date:
                            min_date = file_min
                        if max_date is None or file_max > max_date:
                            max_date = file_max
                except:
                    continue
            
            if min_date and max_date:
                summary['date_ranges']['prices'] = {
                    'start': min_date.isoformat(),
                    'end': max_date.isoformat(),
                    'days': (max_date - min_date).days
                }
        
        # Save summary
        summary_file = self.raw_dir / 'data_collection_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def run_full_collection(self, symbols: List[str], years: int = 10) -> Dict:
        """
        Run complete data collection pipeline.
        
        Args:
            symbols: List of stock symbols
            years: Number of years of historical data
            
        Returns:
            Dictionary with collection results
        """
        logger.info(f"Starting full data collection for {len(symbols)} symbols")
        
        results = {}
        
        # 1. Collect price data
        logger.info("=== Collecting Price Data ===")
        price_data = self.collect_price_data(symbols, years)
        results['price_data'] = len(price_data)
        
        # 2. Collect earnings data
        logger.info("=== Collecting Earnings Data ===")
        earnings_data = self.collect_earnings_data(symbols)
        results['earnings_data'] = len(earnings_data)
        
        # 3. Collect news sentiment
        logger.info("=== Collecting News Sentiment ===")
        sentiment_data = self.collect_news_sentiment(symbols, days=365)
        results['sentiment_data'] = len(sentiment_data)
        
        # 4. Collect fundamental data
        logger.info("=== Collecting Fundamental Data ===")
        fundamental_data = self.collect_fundamental_data(symbols)
        results['fundamental_data'] = len(fundamental_data)
        
        # 5. Collect technical indicators
        logger.info("=== Collecting Technical Indicators ===")
        technical_data = self.collect_technical_indicators(symbols)
        results['technical_data'] = len(technical_data)
        
        # 6. Create summary
        logger.info("=== Creating Data Summary ===")
        summary = self.create_data_summary(symbols)
        results['summary'] = summary
        
        logger.info("Full data collection completed!")
        logger.info(f"Results: {results}")
        
        return results


def main():
    """Main function for data collection script."""
    parser = argparse.ArgumentParser(description='Collect financial data for MHA-DQN research')
    parser.add_argument('--years', type=int, default=10, 
                       help='Number of years of historical data (default: 10)')
    parser.add_argument('--stocks', type=str, default='all',
                       choices=['all', 'large_cap', 'mid_cap', 'small_cap'],
                       help='Stock category to collect (default: all)')
    parser.add_argument('--config', type=str, default='configs/mha_dqn_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Get API key
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("ALPHA_VANTAGE_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Initialize data collector
    collector = DataCollector(config, api_key)
    
    # Get stock list
    symbols = collector.get_stock_list(args.stocks)
    logger.info(f"Collecting data for {len(symbols)} stocks: {symbols}")
    
    # Run collection
    try:
        results = collector.run_full_collection(symbols, args.years)
        
        print("\n" + "="*50)
        print("DATA COLLECTION COMPLETED")
        print("="*50)
        print(f"Symbols processed: {len(symbols)}")
        print(f"Price data: {results.get('price_data', 0)} symbols")
        print(f"Earnings data: {results.get('earnings_data', 0)} symbols")
        print(f"Sentiment data: {results.get('sentiment_data', 0)} symbols")
        print(f"Fundamental data: {results.get('fundamental_data', 0)} symbols")
        print(f"Technical data: {results.get('technical_data', 0)} symbols")
        print(f"\nData saved to: {collector.raw_dir}")
        print("="*50)
        
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
