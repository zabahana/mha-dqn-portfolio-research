#!/usr/bin/env python3
"""
Script to extract text content from Word documents for comparison with LaTeX
"""

import sys
import os
from pathlib import Path

try:
    from docx import Document
except ImportError:
    print("Error: python-docx not installed. Please install it with: pip install python-docx")
    sys.exit(1)

def extract_word_content(docx_path):
    """Extract all text content from a Word document"""
    try:
        doc = Document(docx_path)
        content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text.strip())
        
        return content
    except Exception as e:
        print(f"Error reading Word document: {e}")
        return []

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_word_content.py <path_to_docx_file>")
        sys.exit(1)
    
    docx_path = sys.argv[1]
    if not os.path.exists(docx_path):
        print(f"Error: File {docx_path} not found")
        sys.exit(1)
    
    print(f"Extracting content from: {docx_path}")
    content = extract_word_content(docx_path)
    
    if content:
        print(f"\nExtracted {len(content)} paragraphs:")
        print("=" * 50)
        for i, paragraph in enumerate(content, 1):
            print(f"{i:3d}: {paragraph}")
            print("-" * 30)
    else:
        print("No content extracted")

if __name__ == "__main__":
    main()
