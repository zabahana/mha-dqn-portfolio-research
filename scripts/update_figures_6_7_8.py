#!/usr/bin/env python3
"""
Script to help update figures 6, 7, and 8 in the LaTeX document
"""

import os
from pathlib import Path

def check_figure_files():
    """Check which figure files exist and which need to be updated"""
    figures_dir = Path("paper/figures")
    
    # Figures 6, 7, 8 that need to be updated
    target_figures = {
        "Figure 6": "detailed_architecture.png",
        "Figure 7": "training_flow.png", 
        "Figure 8": "attention_mechanism.png"
    }
    
    print("📊 Figure Status Check:")
    print("=" * 50)
    
    for fig_name, filename in target_figures.items():
        file_path = figures_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {fig_name}: {filename} (exists, {size:,} bytes)")
        else:
            print(f"❌ {fig_name}: {filename} (missing)")
    
    print("\n📋 Instructions to update figures:")
    print("=" * 50)
    print("1. Open your Word document: 'neurips_ieee_publication_ready - ZA.docx'")
    print("2. Find the figures corresponding to:")
    print("   - Figure 6: Detailed MHA-DQN Architecture")
    print("   - Figure 7: Training Flow")
    print("   - Figure 8: Attention Mechanism")
    print("3. Right-click each figure and 'Save as Picture'")
    print("4. Save them in the 'paper/figures/' directory with these exact names:")
    print("   - detailed_architecture.png")
    print("   - training_flow.png")
    print("   - attention_mechanism.png")
    print("\n5. Run this script again to verify the files are in place")
    print("6. Then I can recompile the PDF with the new figures")

def update_latex_if_needed():
    """Update LaTeX file if new figure names are provided"""
    print("\n🔄 Alternative: If you saved the figures with different names,")
    print("   let me know the new filenames and I can update the LaTeX file accordingly.")

if __name__ == "__main__":
    check_figure_files()
    update_latex_if_needed()
