#!/usr/bin/env python3
"""
Batch convert all .drawio files to PNG format
"""

import os
import sys
from pathlib import Path
import subprocess

def check_drawio_installed():
    """Check if draw.io is installed"""
    for cmd in ['drawio', 'draw.io', 'drawio-desktop']:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"draw.io found as '{cmd}': {result.stdout.strip()}")
                return cmd
        except FileNotFoundError:
            continue
    return False

def convert_drawio_to_png(drawio_file, output_file, drawio_cmd, width=1200, height=800):
    """Convert a .drawio file to PNG"""
    try:
        cmd = [
            drawio_cmd,
            '--export',
            '--format', 'png',
            '--width', str(width),
            '--height', str(height),
            '--output', str(output_file),
            str(drawio_file)
        ]
        
        print(f"Converting {drawio_file.name} to {output_file.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully converted {drawio_file.name}")
            return True
        else:
            print(f"❌ Error converting {drawio_file.name}: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running draw.io: {e}")
        return False

def main():
    # Check if draw.io is installed
    drawio_cmd = check_drawio_installed()
    if not drawio_cmd:
        print("❌ Error: draw.io not found. Please install draw.io first.")
        print("You can download it from: https://github.com/jgraph/drawio-desktop/releases")
        print("Or install via:")
        print("  - macOS: brew install --cask drawio")
        print("  - Linux: Download from GitHub releases")
        print("  - Windows: Download installer from GitHub releases")
        sys.exit(1)
    
    # Define the figures directory
    figures_dir = Path("paper/figures")
    if not figures_dir.exists():
        print(f"❌ Error: {figures_dir} directory not found")
        sys.exit(1)
    
    # Find all .drawio files
    drawio_files = list(figures_dir.glob("*.drawio"))
    if not drawio_files:
        print(f"❌ No .drawio files found in {figures_dir}")
        sys.exit(1)
    
    print(f"Found {len(drawio_files)} .drawio files:")
    for file in drawio_files:
        print(f"  - {file.name}")
    
    # Conversion mappings
    conversions = {
        "MHA-DQN Architecture.drawio": {
            "output": "detailed_architecture_from_drawio.png",
            "width": 1400,
            "height": 1000
        },
        "MHA-DQN Training Process Flow.drawio": {
            "output": "training_flow_from_drawio.png", 
            "width": 1200,
            "height": 800
        },
        "Multi-head Attention Mechanism.drawio": {
            "output": "attention_mechanism_from_drawio.png",
            "width": 1000,
            "height": 700
        }
    }
    
    success_count = 0
    total_count = len(drawio_files)
    
    for drawio_file in drawio_files:
        if drawio_file.name in conversions:
            config = conversions[drawio_file.name]
            output_file = figures_dir / config["output"]
            
            success = convert_drawio_to_png(
                drawio_file, 
                output_file, 
                drawio_cmd,
                config["width"], 
                config["height"]
            )
            
            if success:
                success_count += 1
        else:
            # Default conversion for unmapped files
            output_file = drawio_file.with_suffix('.png')
            success = convert_drawio_to_png(
                drawio_file, 
                output_file, 
                drawio_cmd
            )
            if success:
                success_count += 1
    
    print(f"\n📊 Conversion Summary:")
    print(f"  ✅ Successfully converted: {success_count}/{total_count} files")
    
    if success_count == total_count:
        print("🎉 All .drawio files converted successfully!")
        print("\nNext steps:")
        print("1. Review the generated PNG files")
        print("2. Update the LaTeX file to use the new PNG files")
        print("3. Recompile the PDF")
    else:
        print("⚠️  Some conversions failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
