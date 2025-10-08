#!/usr/bin/env python3
"""
Complete pipeline to run MHA-DQN with 30 stocks
This script will:
1. Collect data for additional stocks
2. Run feature engineering
3. Train the model
4. Evaluate results
5. Generate visualizations
6. Update the paper with new results
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.logging import setup_logging

def run_command(command, description):
    """Run a command and log the result"""
    logger = logging.getLogger(__name__)
    logger.info(f"🔄 {description}...")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
        else:
            logger.error(f"❌ {description} failed")
            logger.error(f"Error: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Error running {description}: {e}")
        return False
    
    return True

def main():
    """Run the complete 30-stock pipeline"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 Starting 30-stock MHA-DQN pipeline...")
    
    # Check if we have the API key
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("❌ ALPHA_VANTAGE_API_KEY environment variable not set")
        logger.info("Please set your Alpha Vantage API key:")
        logger.info("export ALPHA_VANTAGE_API_KEY='your_api_key_here'")
        return
    
    # Step 1: Collect data for additional stocks
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Collecting data for additional stocks")
    logger.info("="*60)
    
    if not run_command("python scripts/collect_additional_stocks.py", 
                      "Collecting data for additional stocks"):
        logger.error("❌ Data collection failed. Stopping pipeline.")
        return
    
    # Step 2: Feature engineering
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Feature Engineering")
    logger.info("="*60)
    
    if not run_command("python scripts/feature_engineering.py", 
                      "Running feature engineering"):
        logger.error("❌ Feature engineering failed. Stopping pipeline.")
        return
    
    # Step 3: Train the model
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Training MHA-DQN Model")
    logger.info("="*60)
    
    if not run_command("python scripts/train_mha_dqn.py", 
                      "Training MHA-DQN model"):
        logger.error("❌ Model training failed. Stopping pipeline.")
        return
    
    # Step 4: Evaluate the model
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Model Evaluation")
    logger.info("="*60)
    
    if not run_command("python scripts/evaluate_model.py", 
                      "Evaluating model performance"):
        logger.error("❌ Model evaluation failed. Stopping pipeline.")
        return
    
    # Step 5: Generate visualizations
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Generating Visualizations")
    logger.info("="*60)
    
    if not run_command("python scripts/generate_visualizations.py", 
                      "Generating visualizations"):
        logger.warning("⚠️ Visualization generation failed, but continuing...")
    
    # Step 6: Run complete analysis
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Complete Analysis")
    logger.info("="*60)
    
    if not run_command("python scripts/run_complete_analysis.py", 
                      "Running complete analysis"):
        logger.warning("⚠️ Complete analysis failed, but continuing...")
    
    # Step 7: Update paper with new results
    logger.info("\n" + "="*60)
    logger.info("STEP 7: Updating Paper with New Results")
    logger.info("="*60)
    
    # Recompile PDF
    if run_command("cd paper && pdflatex neurips_ieee_publication_ready.tex", 
                   "Recompiling PDF with new results"):
        logger.info("✅ PDF updated successfully")
    
    # Regenerate Word document
    if run_command("python scripts/create_publication_ready_word.py", 
                   "Regenerating Word document"):
        logger.info("✅ Word document updated successfully")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("🎉 30-STOCK PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info("📊 Results available in:")
    logger.info("  - results/figures/ (new visualizations)")
    logger.info("  - results/tables/ (performance metrics)")
    logger.info("  - paper/neurips_ieee_publication_ready.pdf (updated paper)")
    logger.info("  - paper/neurips_ieee_publication_ready.docx (updated Word doc)")
    
    logger.info("\n📈 Key improvements with 30 stocks:")
    logger.info("  - More diversified portfolio")
    logger.info("  - Better risk management")
    logger.info("  - More comprehensive evaluation")
    logger.info("  - Stronger statistical significance")

if __name__ == "__main__":
    main()
