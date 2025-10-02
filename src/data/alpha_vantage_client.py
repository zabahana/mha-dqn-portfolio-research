"""
Alpha Vantage API Client for Financial Data Collection
Research Project: Multi-Head Attention DQN for Portfolio Optimization
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.techindicators import TechIndicators

logger = logging.getLogger(__name__)


class AlphaVantageClient:
    """
    Comprehensive Alpha Vantage API client for financial data collection.
    
    Features:
    - Daily/Intraday price data
    - Earnings call transcripts and sentiment
    - Fundamental data (P/E, P/B, etc.)
    - News sentiment analysis
    - Technical indicators
    - Rate limiting and error handling
    """
    
    def __init__(self, api_key: Optional[str] = None, premium: bool = True):
        """
        Initialize Alpha Vantage client.
        
        Args:
            api_key: Alpha Vantage API key (if None, reads from environment)
            premium: Whether using premium subscription (higher rate limits)
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not provided")
            
        self.premium = premium
        self.base_url = "https://www.alphavantage.co/query"
        
        # Rate limiting (premium: 75 calls/min, free: 5 calls/min)
        self.rate_limit = 75 if premium else 5
        self.call_timestamps = []
        
        # Initialize API clients
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.fd = FundamentalData(key=self.api_key, output_format='pandas')
        self.ti = TechIndicators(key=self.api_key, output_format='pandas')
        self.sp = SectorPerformances(key=self.api_key, output_format='pandas')
        
        logger.info(f"Alpha Vantage client initialized (Premium: {premium})")
    
    def _rate_limit_check(self):
        """Ensure we don't exceed API rate limits."""
        now = time.time()
        # Remove timestamps older than 1 minute
        self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < 60]
        
        if len(self.call_timestamps) >= self.rate_limit:
            sleep_time = 60 - (now - self.call_timestamps[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self.call_timestamps.append(now)
    
    def get_daily_prices(self, symbol: str, outputsize: str = 'full') -> pd.DataFrame:
        """
        Get daily price data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            outputsize: 'compact' (100 days) or 'full' (20+ years)
            
        Returns:
            DataFrame with OHLCV data
        """
        self._rate_limit_check()
        
        try:
            data, meta_data = self.ts.get_daily_adjusted(symbol=symbol, outputsize=outputsize)
            
            # Clean column names
            data.columns = [col.split('. ')[1] for col in data.columns]
            data.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
            
            # Sort by date (oldest first)
            data = data.sort_index()
            
            logger.info(f"Retrieved {len(data)} days of price data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_earnings_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get earnings data including quarterly and annual reports.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (quarterly_earnings, annual_earnings) DataFrames
        """
        self._rate_limit_check()
        
        try:
            quarterly, annual = self.fd.get_earnings(symbol=symbol)
            
            # Convert to datetime
            if not quarterly.empty:
                quarterly.index = pd.to_datetime(quarterly.index)
                quarterly = quarterly.sort_index()
                
            if not annual.empty:
                annual.index = pd.to_datetime(annual.index)
                annual = annual.sort_index()
            
            logger.info(f"Retrieved earnings data for {symbol}: {len(quarterly)} quarterly, {len(annual)} annual")
            return quarterly, annual
            
        except Exception as e:
            logger.error(f"Error fetching earnings data for {symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_earnings_calendar(self, horizon: str = '3month') -> pd.DataFrame:
        """
        Get upcoming earnings calendar.
        
        Args:
            horizon: '3month', '6month', or '12month'
            
        Returns:
            DataFrame with upcoming earnings dates
        """
        self._rate_limit_check()
        
        params = {
            'function': 'EARNINGS_CALENDAR',
            'horizon': horizon,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                # Parse CSV response
                from io import StringIO
                data = pd.read_csv(StringIO(response.text))
                
                # Convert date column
                data['reportDate'] = pd.to_datetime(data['reportDate'])
                
                logger.info(f"Retrieved earnings calendar: {len(data)} upcoming earnings")
                return data
            else:
                logger.error(f"Error fetching earnings calendar: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {e}")
            return pd.DataFrame()
    
    def get_news_sentiment(self, tickers: Union[str, List[str]], 
                          time_from: Optional[str] = None,
                          time_to: Optional[str] = None,
                          limit: int = 1000) -> pd.DataFrame:
        """
        Get news sentiment data for given tickers.
        
        Args:
            tickers: Single ticker or list of tickers
            time_from: Start date (YYYYMMDDTHHMM format)
            time_to: End date (YYYYMMDDTHHMM format)
            limit: Maximum number of articles (1-1000)
            
        Returns:
            DataFrame with news sentiment data
        """
        self._rate_limit_check()
        
        if isinstance(tickers, list):
            tickers = ','.join(tickers)
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': tickers,
            'apikey': self.api_key,
            'limit': limit
        }
        
        if time_from:
            params['time_from'] = time_from
        if time_to:
            params['time_to'] = time_to
        
        try:
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'feed' in data:
                    articles = []
                    for article in data['feed']:
                        article_data = {
                            'time_published': article.get('time_published'),
                            'title': article.get('title'),
                            'summary': article.get('summary'),
                            'overall_sentiment_score': article.get('overall_sentiment_score'),
                            'overall_sentiment_label': article.get('overall_sentiment_label'),
                        }
                        
                        # Add ticker-specific sentiment
                        if 'ticker_sentiment' in article:
                            for ticker_sent in article['ticker_sentiment']:
                                ticker = ticker_sent.get('ticker')
                                article_data[f'{ticker}_sentiment_score'] = ticker_sent.get('ticker_sentiment_score')
                                article_data[f'{ticker}_sentiment_label'] = ticker_sent.get('ticker_sentiment_label')
                        
                        articles.append(article_data)
                    
                    df = pd.DataFrame(articles)
                    if not df.empty:
                        df['time_published'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S')
                        df = df.sort_values('time_published')
                    
                    logger.info(f"Retrieved {len(df)} news articles with sentiment")
                    return df
                else:
                    logger.warning("No news data found in response")
                    return pd.DataFrame()
            else:
                logger.error(f"Error fetching news sentiment: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return pd.DataFrame()
    
    def get_company_overview(self, symbol: str) -> Dict:
        """
        Get company overview and fundamental data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company overview data
        """
        self._rate_limit_check()
        
        try:
            data, _ = self.fd.get_company_overview(symbol=symbol)
            
            if not data.empty:
                # Convert to dictionary
                overview = data.iloc[0].to_dict()
                logger.info(f"Retrieved company overview for {symbol}")
                return overview
            else:
                logger.warning(f"No company overview data for {symbol}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching company overview for {symbol}: {e}")
            return {}
    
    def get_technical_indicators(self, symbol: str, indicator: str, 
                               interval: str = 'daily', **kwargs) -> pd.DataFrame:
        """
        Get technical indicators for a symbol.
        
        Args:
            symbol: Stock symbol
            indicator: Technical indicator name (e.g., 'SMA', 'RSI', 'MACD')
            interval: Time interval ('daily', 'weekly', 'monthly')
            **kwargs: Additional parameters for the indicator
            
        Returns:
            DataFrame with technical indicator data
        """
        self._rate_limit_check()
        
        try:
            # Map indicator names to methods
            indicator_methods = {
                'SMA': self.ti.get_sma,
                'EMA': self.ti.get_ema,
                'RSI': self.ti.get_rsi,
                'MACD': self.ti.get_macd,
                'BBANDS': self.ti.get_bbands,
                'ATR': self.ti.get_atr,
                'ADX': self.ti.get_adx,
                'STOCH': self.ti.get_stoch,
            }
            
            if indicator.upper() not in indicator_methods:
                raise ValueError(f"Unsupported indicator: {indicator}")
            
            method = indicator_methods[indicator.upper()]
            data, meta_data = method(symbol=symbol, interval=interval, **kwargs)
            
            # Sort by date
            data = data.sort_index()
            
            logger.info(f"Retrieved {indicator} data for {symbol}: {len(data)} points")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {indicator} for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_earnings_transcript_sentiment(self, symbol: str, year: int, quarter: int) -> Dict:
        """
        Get earnings call transcript and sentiment analysis.
        Note: This is a premium feature and may require special API access.
        
        Args:
            symbol: Stock symbol
            year: Year of earnings call
            quarter: Quarter (1, 2, 3, 4)
            
        Returns:
            Dictionary with transcript and sentiment data
        """
        # This would require special API access or web scraping
        # For now, return placeholder structure
        logger.warning("Earnings transcript sentiment not yet implemented")
        return {
            'symbol': symbol,
            'year': year,
            'quarter': quarter,
            'transcript': '',
            'sentiment_score': 0.0,
            'key_topics': [],
            'management_tone': 'neutral'
        }
    
    def batch_collect_data(self, symbols: List[str], 
                          start_date: str, end_date: str,
                          include_earnings: bool = True,
                          include_news: bool = True,
                          include_fundamentals: bool = True) -> Dict[str, Dict]:
        """
        Collect comprehensive data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_earnings: Whether to include earnings data
            include_news: Whether to include news sentiment
            include_fundamentals: Whether to include fundamental data
            
        Returns:
            Dictionary with all collected data
        """
        logger.info(f"Starting batch data collection for {len(symbols)} symbols")
        
        all_data = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
            
            symbol_data = {}
            
            # Price data
            price_data = self.get_daily_prices(symbol, outputsize='full')
            if not price_data.empty:
                # Filter by date range
                mask = (price_data.index >= start_date) & (price_data.index <= end_date)
                symbol_data['prices'] = price_data.loc[mask]
            
            # Earnings data
            if include_earnings:
                quarterly, annual = self.get_earnings_data(symbol)
                symbol_data['earnings_quarterly'] = quarterly
                symbol_data['earnings_annual'] = annual
            
            # Company overview
            if include_fundamentals:
                overview = self.get_company_overview(symbol)
                symbol_data['overview'] = overview
            
            # News sentiment (last 30 days as example)
            if include_news:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                start_dt = end_dt - timedelta(days=30)
                
                time_from = start_dt.strftime('%Y%m%dT0000')
                time_to = end_dt.strftime('%Y%m%dT2359')
                
                news_data = self.get_news_sentiment(symbol, time_from=time_from, time_to=time_to)
                symbol_data['news_sentiment'] = news_data
            
            all_data[symbol] = symbol_data
            
            # Small delay between symbols to be respectful
            time.sleep(1)
        
        logger.info("Batch data collection completed")
        return all_data


def main():
    """Example usage of AlphaVantageClient."""
    # Initialize client
    client = AlphaVantageClient(premium=True)
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Collect data
    data = client.batch_collect_data(
        symbols=test_symbols,
        start_date='2023-01-01',
        end_date='2024-01-01',
        include_earnings=True,
        include_news=True,
        include_fundamentals=True
    )
    
    # Print summary
    for symbol, symbol_data in data.items():
        print(f"\n{symbol}:")
        for data_type, df in symbol_data.items():
            if isinstance(df, pd.DataFrame):
                print(f"  {data_type}: {len(df)} records")
            else:
                print(f"  {data_type}: {type(df)}")


if __name__ == "__main__":
    main()
