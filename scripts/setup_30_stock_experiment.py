#!/usr/bin/env python3
"""
Setup script for 30-stock MHA-DQN experiment
This script helps you set up the environment and run the 30-stock pipeline
"""

import os
import sys
from pathlib import Path

def main():
    """Setup the 30-stock experiment"""
    
    print("🚀 Setting up 30-stock MHA-DQN experiment...")
    print("="*60)
    
    # Check for API key
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("❌ Alpha Vantage API key not found!")
        print("\n📋 To get started, you need to:")
        print("1. Get a free API key from: https://www.alphavantage.co/support/#api-key")
        print("2. Set the environment variable:")
        print("   export ALPHA_VANTAGE_API_KEY='your_api_key_here'")
        print("\n💡 Or create a .env file in the project root with:")
        print("   ALPHA_VANTAGE_API_KEY=your_api_key_here")
        print("\n🔄 After setting up the API key, run:")
        print("   python scripts/run_30_stock_pipeline.py")
        return
    
    print("✅ Alpha Vantage API key found!")
    
    # Check current data
    data_dir = Path("data/raw")
    existing_stocks = []
    if data_dir.exists():
        for file in data_dir.glob("*_price_data.csv"):
            stock = file.name.replace("_price_data.csv", "")
            existing_stocks.append(stock)
    
    print(f"\n📊 Current data status:")
    print(f"   Existing stocks: {len(existing_stocks)}")
    print(f"   Stocks: {existing_stocks}")
    
    # Define target stocks
    target_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "UNH", "JNJ", "BRK.B",
        "JPM", "V", "PG", "MA", "HD", "BAC", "DIS", "KO", "CVX", "MRK",
        "PFE", "INTC", "CSCO", "VZ", "T", "CMCSA", "NFLX", "ADBE", "CRM", "ORCL"
    ]
    
    missing_stocks = [stock for stock in target_stocks if stock not in existing_stocks]
    
    if missing_stocks:
        print(f"\n📥 Missing stocks ({len(missing_stocks)}): {missing_stocks}")
        print("\n🔄 To collect data for missing stocks, run:")
        print("   python scripts/collect_additional_stocks.py")
    else:
        print("\n✅ All 30 stocks have data available!")
        print("\n🚀 Ready to run the complete pipeline:")
        print("   python scripts/run_30_stock_pipeline.py")
    
    print("\n📋 Pipeline steps:")
    print("1. Collect data for additional stocks")
    print("2. Run feature engineering")
    print("3. Train MHA-DQN model")
    print("4. Evaluate model performance")
    print("5. Generate visualizations")
    print("6. Update paper with new results")
    
    print("\n⏱️ Estimated time: 2-4 hours (depending on API rate limits)")
    print("💡 The pipeline will automatically handle all steps")

if __name__ == "__main__":
    main()
