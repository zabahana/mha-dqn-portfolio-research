#!/usr/bin/env python3
"""
Script to collect data for the additional 20 stocks to expand from 9 to 30 stocks
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from scripts.collect_data import DataCollector
from utils.config import load_config
from utils.logging import setup_logging

def main():
    """Collect data for additional stocks to expand to 30 stocks"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting data collection for additional stocks...")
    
    # Load configuration
    config_path = Path("configs/mha_dqn_config.yaml")
    config = load_config(config_path)
    
    # Get API key
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("ALPHA_VANTAGE_API_KEY environment variable not set")
        return
    
    # Initialize data collector
    collector = DataCollector(config, api_key)
    
    # Define the additional stocks we need to collect
    # Current stocks: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, UNH, JNJ, BRK.B (10 stocks)
    # We need 20 more to reach 30 total
    additional_stocks = [
        "JPM", "V", "PG", "MA", "HD", "BAC", "DIS", "KO", "CVX", "MRK",
        "PFE", "INTC", "CSCO", "VZ", "T", "CMCSA", "NFLX", "ADBE", "CRM", "ORCL"
    ]
    
    logger.info(f"Collecting data for {len(additional_stocks)} additional stocks: {additional_stocks}")
    
    # Collect data for each additional stock
    success_count = 0
    failed_stocks = []
    
    for stock in additional_stocks:
        try:
            logger.info(f"Collecting data for {stock}...")
            
            # Collect price data
            price_data = collector.collect_price_data(stock, years=5)
            if price_data is not None and not price_data.empty:
                logger.info(f"✅ Successfully collected price data for {stock}")
                
                # Save price data
                price_path = Path(f"data/raw/{stock}_price_data.csv")
                price_data.to_csv(price_path, index=True)
                
                # Collect fundamental data
                try:
                    fundamental_data = collector.collect_fundamental_data(stock)
                    if fundamental_data is not None and not fundamental_data.empty:
                        fundamental_path = Path(f"data/raw/{stock}_fundamental_data.csv")
                        fundamental_data.to_csv(fundamental_path, index=True)
                        logger.info(f"✅ Successfully collected fundamental data for {stock}")
                    else:
                        logger.warning(f"⚠️ No fundamental data available for {stock}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to collect fundamental data for {stock}: {e}")
                
                # Collect earnings data
                try:
                    earnings_data = collector.collect_earnings_data(stock)
                    if earnings_data is not None and not earnings_data.empty:
                        earnings_path = Path(f"data/raw/{stock}_earnings_data.csv")
                        earnings_data.to_csv(earnings_path, index=True)
                        logger.info(f"✅ Successfully collected earnings data for {stock}")
                    else:
                        logger.warning(f"⚠️ No earnings data available for {stock}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to collect earnings data for {stock}: {e}")
                
                success_count += 1
                
            else:
                logger.error(f"❌ Failed to collect price data for {stock}")
                failed_stocks.append(stock)
                
        except Exception as e:
            logger.error(f"❌ Error collecting data for {stock}: {e}")
            failed_stocks.append(stock)
        
        # Add delay to respect API rate limits
        import time
        time.sleep(12)  # Alpha Vantage free tier: 5 calls per minute
    
    # Summary
    logger.info(f"\n📊 Data Collection Summary:")
    logger.info(f"✅ Successfully collected: {success_count}/{len(additional_stocks)} stocks")
    
    if failed_stocks:
        logger.warning(f"❌ Failed stocks: {failed_stocks}")
    
    if success_count == len(additional_stocks):
        logger.info("🎉 All additional stocks collected successfully!")
        logger.info("Next steps:")
        logger.info("1. Run feature engineering: python scripts/feature_engineering.py")
        logger.info("2. Train the model: python scripts/train_mha_dqn.py")
        logger.info("3. Evaluate results: python scripts/evaluate_model.py")
    else:
        logger.warning("⚠️ Some stocks failed. Please check the logs and retry failed stocks.")

if __name__ == "__main__":
    main()
