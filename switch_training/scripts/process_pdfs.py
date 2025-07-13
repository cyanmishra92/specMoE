#!/usr/bin/env python3
"""
PDF Processing Pipeline for Switch Transformer Training

This script processes research papers (PDFs) into training data for Switch Transformer.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re
from datetime import datetime

# PDF processing
import PyPDF2
import fitz  # pymupdf
from pdfplumber import PDF

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import tiktoken
from transformers import AutoTokenizer

# Data handling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDFs into training data for Switch Transformer"""
    
    def __init__(self, raw_pdf_dir: str, output_dir: str, 
                 tokenizer_name: str = "gpt2", max_seq_length: int = 512):
        self.raw_pdf_dir = Path(raw_pdf_dir)
        self.output_dir = Path(output_dir)
        self.max_seq_length = max_seq_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'processed').mkdir(parents=True, exist_ok=True)
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using multiple methods for robustness"""
        logger.info(f"Processing {pdf_path.name}")
        
        text = ""
        
        # Method 1: PyMuPDF (fitz) - best for most PDFs
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            logger.info(f"✅ Extracted {len(text)} characters using PyMuPDF")
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")
            
            # Method 2: pdfplumber - better for complex layouts
            try:
                with PDF.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                logger.info(f"✅ Extracted {len(text)} characters using pdfplumber")
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")
                
                # Method 3: PyPDF2 - fallback
                try:
                    with open(pdf_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        for page in reader.pages:
                            text += page.extract_text()
                    logger.info(f"✅ Extracted {len(text)} characters using PyPDF2")
                except Exception as e:
                    logger.error(f"All PDF extraction methods failed for {pdf_path}: {e}")
                    return ""
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text.strip():
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)  # Standalone page numbers
        text = re.sub(r'Page \d+', '', text)
        
        # Remove URLs and emails
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Clean up special characters but preserve punctuation
        text = re.sub(r'[^\w\s.,;:!?()\[\]{}"\'\\-]', ' ', text)
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def segment_text(self, text: str, overlap_sentences: int = 2) -> List[str]:
        """Segment text into overlapping chunks suitable for training"""
        if not text.strip():
            return []
        
        # Split into sentences
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return [text] if text.strip() else []
        
        segments = []
        
        # Calculate tokens per segment (leave room for special tokens)
        target_tokens = self.max_seq_length - 10
        
        current_segment = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # If adding this sentence would exceed limit, save current segment
            if current_tokens + sentence_tokens > target_tokens and current_segment:
                segment_text = ' '.join(current_segment)
                if len(segment_text.strip()) > 50:  # Minimum segment length
                    segments.append(segment_text)
                
                # Start new segment with overlap
                if len(current_segment) > overlap_sentences:
                    current_segment = current_segment[-overlap_sentences:]
                    current_tokens = sum(len(self.tokenizer.encode(s)) for s in current_segment)
                else:
                    current_segment = []
                    current_tokens = 0
            
            current_segment.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final segment
        if current_segment:
            segment_text = ' '.join(current_segment)
            if len(segment_text.strip()) > 50:
                segments.append(segment_text)
        
        return segments
    
    def process_all_pdfs(self) -> Dict[str, List[str]]:
        """Process all PDFs in the raw_pdf directory"""
        logger.info(f"Processing PDFs from {self.raw_pdf_dir}")
        
        pdf_files = list(self.raw_pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in {self.raw_pdf_dir}")
            return {}
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_segments = {}
        total_segments = 0
        
        for pdf_path in pdf_files:
            try:
                # Extract and clean text
                raw_text = self.extract_text_from_pdf(pdf_path)
                if not raw_text.strip():
                    logger.warning(f"No text extracted from {pdf_path.name}")
                    continue
                
                cleaned_text = self.clean_text(raw_text)
                if len(cleaned_text) < 100:
                    logger.warning(f"Insufficient text in {pdf_path.name} ({len(cleaned_text)} chars)")
                    continue
                
                # Segment text
                segments = self.segment_text(cleaned_text)
                if not segments:
                    logger.warning(f"No valid segments created from {pdf_path.name}")
                    continue
                
                all_segments[pdf_path.name] = segments
                total_segments += len(segments)
                
                logger.info(f"✅ {pdf_path.name}: {len(segments)} segments")
                
                # Save processed text for inspection
                processed_file = self.output_dir / 'processed' / f"{pdf_path.stem}.txt"
                with open(processed_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== ORIGINAL PDF: {pdf_path.name} ===\\n\\n")
                    f.write(f"Raw text length: {len(raw_text)} characters\\n")
                    f.write(f"Cleaned text length: {len(cleaned_text)} characters\\n")
                    f.write(f"Number of segments: {len(segments)}\\n\\n")
                    f.write("=== CLEANED TEXT ===\\n\\n")
                    f.write(cleaned_text)
                    f.write("\\n\\n=== SEGMENTS ===\\n\\n")
                    for i, segment in enumerate(segments):
                        f.write(f"--- Segment {i+1} ---\\n{segment}\\n\\n")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue
        
        logger.info(f"Processing complete: {total_segments} total segments from {len(all_segments)} PDFs")
        return all_segments
    
    def create_training_splits(self, all_segments: Dict[str, List[str]], 
                             train_ratio: float = 0.8, val_ratio: float = 0.1,
                             test_ratio: float = 0.1) -> None:
        """Create train/val/test splits from processed segments"""
        logger.info("Creating training splits...")
        
        # Flatten all segments with source tracking
        flat_segments = []
        for pdf_name, segments in all_segments.items():
            for segment in segments:
                flat_segments.append({
                    'text': segment,
                    'source': pdf_name,
                    'length': len(segment),
                    'tokens': len(self.tokenizer.encode(segment))
                })
        
        if not flat_segments:
            logger.error("No segments available for splitting")
            return
        
        logger.info(f"Total segments for splitting: {len(flat_segments)}")
        
        # Shuffle segments
        np.random.seed(42)
        np.random.shuffle(flat_segments)
        
        # Calculate split sizes
        total = len(flat_segments)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        test_size = total - train_size - val_size
        
        # Create splits
        train_segments = flat_segments[:train_size]
        val_segments = flat_segments[train_size:train_size + val_size]
        test_segments = flat_segments[train_size + val_size:]
        
        logger.info(f"Split sizes - Train: {len(train_segments)}, Val: {len(val_segments)}, Test: {len(test_segments)}")
        
        # Save splits
        splits = {
            'train': train_segments,
            'val': val_segments, 
            'test': test_segments
        }
        
        for split_name, segments in splits.items():
            # Save as JSON
            output_file = self.output_dir / split_name / f"{split_name}_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
            
            # Save as text file for inspection
            text_file = self.output_dir / split_name / f"{split_name}_data.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments):
                    f.write(f"=== Sample {i+1} ===\\n")
                    f.write(f"Source: {segment['source']}\\n")
                    f.write(f"Length: {segment['length']} chars, {segment['tokens']} tokens\\n")
                    f.write(f"Text: {segment['text']}\\n\\n")
            
            logger.info(f"✅ Saved {split_name} split: {len(segments)} segments")
        
        # Save processing statistics
        stats = {
            'processing_date': datetime.now().isoformat(),
            'total_pdfs': len(all_segments),
            'total_segments': len(flat_segments),
            'splits': {
                'train': len(train_segments),
                'val': len(val_segments),
                'test': len(test_segments)
            },
            'tokenizer': self.tokenizer.name_or_path,
            'max_seq_length': self.max_seq_length,
            'avg_segment_length': np.mean([s['length'] for s in flat_segments]),
            'avg_tokens_per_segment': np.mean([s['tokens'] for s in flat_segments]),
            'pdf_sources': list(all_segments.keys())
        }
        
        stats_file = self.output_dir / 'processing_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✅ Processing statistics saved to {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="Process PDFs for Switch Transformer training")
    parser.add_argument("--raw_pdf_dir", type=str, 
                       default="../data/raw_pdfs",
                       help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str,
                       default="../data",
                       help="Output directory for processed data")
    parser.add_argument("--tokenizer", type=str,
                       default="gpt2",
                       help="Tokenizer to use for segmentation")
    parser.add_argument("--max_seq_length", type=int,
                       default=512,
                       help="Maximum sequence length for segments")
    parser.add_argument("--train_ratio", type=float,
                       default=0.8,
                       help="Training data ratio")
    parser.add_argument("--val_ratio", type=float,
                       default=0.1,
                       help="Validation data ratio")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.train_ratio + args.val_ratio >= 1.0:
        logger.error("Train + validation ratios must be < 1.0")
        return
    
    # Initialize processor
    processor = PDFProcessor(
        raw_pdf_dir=args.raw_pdf_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length
    )
    
    # Process PDFs
    all_segments = processor.process_all_pdfs()
    
    if not all_segments:
        logger.error("No segments were created. Please check your PDF files.")
        return
    
    # Create splits
    processor.create_training_splits(
        all_segments,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio
    )
    
    logger.info("✅ PDF processing pipeline completed successfully!")

if __name__ == "__main__":
    main()