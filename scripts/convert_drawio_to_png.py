#!/usr/bin/env python3
"""
Script to convert .drawio files to PNG format for use in LaTeX documents
"""

import os
import sys
import subprocess
from pathlib import Path

def check_drawio_installed():
    """Check if draw.io is installed"""
    try:
        result = subprocess.run(['drawio', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"draw.io found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    # Try alternative command names
    for cmd in ['draw.io', 'drawio-desktop']:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"draw.io found as '{cmd}': {result.stdout.strip()}")
                return cmd
        except FileNotFoundError:
            continue
    
    return False

def convert_drawio_to_png(drawio_file, output_file, width=1200, height=800):
    """Convert a .drawio file to PNG"""
    drawio_cmd = check_drawio_installed()
    if not drawio_cmd:
        print("Error: draw.io not found. Please install draw.io first.")
        print("You can download it from: https://github.com/jgraph/drawio-desktop/releases")
        return False
    
    if isinstance(drawio_cmd, bool):
        drawio_cmd = 'drawio'
    
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
        
        print(f"Converting {drawio_file} to {output_file}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully converted {drawio_file} to {output_file}")
            return True
        else:
            print(f"❌ Error converting {drawio_file}: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running draw.io: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_drawio_to_png.py <drawio_file> [output_file] [width] [height]")
        print("Example: python convert_drawio_to_png.py figures/architecture.drawio figures/architecture.png 1200 800")
        sys.exit(1)
    
    drawio_file = Path(sys.argv[1])
    if not drawio_file.exists():
        print(f"Error: File {drawio_file} not found")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        output_file = drawio_file.with_suffix('.png')
    
    width = int(sys.argv[3]) if len(sys.argv) >= 4 else 1200
    height = int(sys.argv[4]) if len(sys.argv) >= 5 else 800
    
    success = convert_drawio_to_png(drawio_file, output_file, width, height)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
