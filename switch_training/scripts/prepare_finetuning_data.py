#!/usr/bin/env python3
"""
Fine-tuning Data Preparation for Switch Transformer

Optimized for domain-specific fine-tuning with limited data (10-12 papers).
Includes data augmentation techniques to maximize training effectiveness.
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
import random

# PDF processing
import PyPDF2
import fitz  # pymupdf
from pdfplumber import PDF

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
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

class FineTuningDataProcessor:
    """Process PDFs for Switch Transformer fine-tuning with data augmentation"""
    
    def __init__(self, raw_pdf_dir: str, output_dir: str, 
                 tokenizer_name: str = "google/switch-base-8", 
                 max_seq_length: int = 512,
                 augmentation_factor: int = 3):
        self.raw_pdf_dir = Path(raw_pdf_dir)
        self.output_dir = Path(output_dir)
        self.max_seq_length = max_seq_length
        self.augmentation_factor = augmentation_factor
        
        # Initialize tokenizer (Switch Transformer compatible)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            logger.warning(f"Switch tokenizer not found, falling back to T5")
            self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'processed').mkdir(parents=True, exist_ok=True)
        
        # Download NLTK data (handle both old and new versions)
        nltk_downloads = ['punkt', 'punkt_tab']
        for resource in nltk_downloads:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download {resource}: {e}")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF with academic paper optimization"""
        logger.info(f"Processing {pdf_path.name}")
        
        text = ""
        
        # Try PyMuPDF first - best for academic papers
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text blocks to preserve structure
                blocks = page.get_text("dict")["blocks"]
                page_text = ""
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                page_text += span["text"] + " "
                        page_text += "\\n"
                
                text += page_text
            doc.close()
            logger.info(f"✅ Extracted {len(text)} characters using PyMuPDF")
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}, trying fallback methods")
            
            # Fallback to pdfplumber
            try:
                with PDF.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\\n"
                logger.info(f"✅ Extracted {len(text)} characters using pdfplumber")
            except Exception as e:
                logger.error(f"All extraction methods failed for {pdf_path}: {e}")
                return ""
        
        return text
    
    def clean_academic_text(self, text: str) -> str:
        """Clean text with academic paper specific optimizations"""
        if not text.strip():
            return ""
        
        # Remove common academic artifacts
        text = re.sub(r'\\b(?:Figure|Table|Equation)\\s+\\d+', '', text)  # Figure/Table references
        text = re.sub(r'\\[\\d+\\]', '', text)  # Citation numbers
        text = re.sub(r'\\b(?:et\\s+al\\.?)', 'et al.', text)  # Normalize et al
        
        # Remove page headers/footers
        text = re.sub(r'^.*?(?:Abstract|Introduction|ABSTRACT|INTRODUCTION)', 'Abstract', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'References?\\s*$.*', '', text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        # Clean spacing and formatting
        text = re.sub(r'\\s+', ' ', text)
        text = re.sub(r'\\n+', '\\n', text)
        text = text.strip()
        
        return text
    
    def segment_for_finetuning(self, text: str) -> List[Dict[str, str]]:
        """Create segments optimized for fine-tuning with metadata"""
        if not text.strip():
            return []
        
        # Split into sentences
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return []
        
        segments = []
        
        # Strategy 1: Paragraph-based segments (academic structure)
        paragraphs = text.split('\\n\\n')
        for para in paragraphs:
            if len(para.strip()) > 100:  # Minimum paragraph length
                tokens = len(self.tokenizer.encode(para))
                if tokens <= self.max_seq_length - 10:
                    segments.append({
                        'text': para.strip(),
                        'type': 'paragraph',
                        'tokens': tokens
                    })
        
        # Strategy 2: Sliding window segments for better coverage
        window_size = 3  # sentences per window
        overlap = 1      # sentence overlap
        
        for i in range(0, len(sentences) - window_size + 1, window_size - overlap):
            window_text = ' '.join(sentences[i:i + window_size])
            tokens = len(self.tokenizer.encode(window_text))
            
            if tokens <= self.max_seq_length - 10 and len(window_text) > 50:
                segments.append({
                    'text': window_text,
                    'type': 'sliding_window',
                    'tokens': tokens
                })
        
        # Strategy 3: Section-based segments (if sections are detectable)
        section_patterns = [r'\\n(?:Abstract|Introduction|Method|Results|Discussion|Conclusion)',
                          r'\\n\\d+\\.\\s+[A-Z][^\\n]*',  # Numbered sections
                          r'\\n[A-Z][A-Z\\s]+\\n']        # ALL CAPS sections
        
        for pattern in section_patterns:
            sections = re.split(pattern, text, flags=re.IGNORECASE)
            for section in sections[1:]:  # Skip first empty split
                if len(section.strip()) > 200:  # Minimum section length
                    # Truncate if too long
                    words = section.split()
                    if len(words) > 400:  # Approximate token limit
                        section = ' '.join(words[:400])
                    
                    tokens = len(self.tokenizer.encode(section))
                    if tokens <= self.max_seq_length - 10:
                        segments.append({
                            'text': section.strip(),
                            'type': 'section',
                            'tokens': tokens
                        })
        
        return segments
    
    def augment_data(self, segments: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply data augmentation techniques for fine-tuning"""
        augmented = segments.copy()  # Keep originals
        
        for segment in segments:
            text = segment['text']
            
            # Augmentation 1: Sentence shuffling (within paragraph)
            sentences = sent_tokenize(text)
            if len(sentences) >= 3:
                shuffled_sentences = sentences.copy()
                random.shuffle(shuffled_sentences)
                shuffled_text = ' '.join(shuffled_sentences)
                
                augmented.append({
                    'text': shuffled_text,
                    'type': f"{segment['type']}_shuffled",
                    'tokens': len(self.tokenizer.encode(shuffled_text)),
                    'augmented': True
                })
            
            # Augmentation 2: Partial text (first/last parts)
            if len(sentences) >= 4:
                # First 70% of sentences
                partial_sentences = sentences[:int(len(sentences) * 0.7)]
                partial_text = ' '.join(partial_sentences)
                
                if len(partial_text) > 50:
                    augmented.append({
                        'text': partial_text,
                        'type': f"{segment['type']}_partial",
                        'tokens': len(self.tokenizer.encode(partial_text)),
                        'augmented': True
                    })
        
        logger.info(f"Augmented {len(segments)} segments to {len(augmented)} segments")
        return augmented
    
    def process_all_pdfs(self) -> Dict[str, List[Dict[str, str]]]:
        """Process all PDFs for fine-tuning"""
        logger.info(f"Processing PDFs for fine-tuning from {self.raw_pdf_dir}")
        
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
                
                cleaned_text = self.clean_academic_text(raw_text)
                if len(cleaned_text) < 500:  # Minimum for academic paper
                    logger.warning(f"Insufficient text in {pdf_path.name} ({len(cleaned_text)} chars)")
                    continue
                
                # Create base segments
                base_segments = self.segment_for_finetuning(cleaned_text)
                if not base_segments:
                    logger.warning(f"No valid segments from {pdf_path.name}")
                    continue
                
                # Apply augmentation
                augmented_segments = self.augment_data(base_segments)
                
                all_segments[pdf_path.name] = augmented_segments
                total_segments += len(augmented_segments)
                
                logger.info(f"✅ {pdf_path.name}: {len(base_segments)} → {len(augmented_segments)} segments")
                
                # Save processed text
                processed_file = self.output_dir / 'processed' / f"{pdf_path.stem}_processed.json"
                with open(processed_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'source': pdf_path.name,
                        'raw_length': len(raw_text),
                        'cleaned_length': len(cleaned_text),
                        'base_segments': len(base_segments),
                        'augmented_segments': len(augmented_segments),
                        'segments': augmented_segments
                    }, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue
        
        logger.info(f"Processing complete: {total_segments} total segments from {len(all_segments)} PDFs")
        return all_segments
    
    def create_finetuning_splits(self, all_segments: Dict[str, List[Dict[str, str]]]) -> None:
        """Create balanced splits for fine-tuning"""
        logger.info("Creating fine-tuning splits...")
        
        # Flatten with source tracking
        flat_segments = []
        for pdf_name, segments in all_segments.items():
            for segment in segments:
                flat_segments.append({
                    **segment,
                    'source': pdf_name,
                    'length': len(segment['text'])
                })
        
        if not flat_segments:
            logger.error("No segments for splitting")
            return
        
        logger.info(f"Total segments: {len(flat_segments)}")
        
        # Stratified split to ensure representation from all papers
        sources = [seg['source'] for seg in flat_segments]
        
        # Use stratified split to maintain source distribution
        train_segments, temp_segments = train_test_split(
            flat_segments, test_size=0.2, random_state=42,
            stratify=sources if len(set(sources)) > 1 else None
        )
        
        val_segments, test_segments = train_test_split(
            temp_segments, test_size=0.5, random_state=42
        )
        
        splits = {
            'train': train_segments,
            'val': val_segments,
            'test': test_segments
        }
        
        for split_name, segments in splits.items():
            # Save for training
            output_file = self.output_dir / split_name / f"finetuning_{split_name}.json"
            
            # Format for HuggingFace datasets
            formatted_data = []
            for segment in segments:
                formatted_data.append({
                    'text': segment['text'],
                    'source': segment['source'],
                    'type': segment['type'],
                    'tokens': segment['tokens'],
                    'augmented': segment.get('augmented', False)
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ {split_name}: {len(segments)} segments")
        
        # Save comprehensive statistics
        stats = {
            'processing_date': datetime.now().isoformat(),
            'num_pdfs': len(all_segments),
            'total_segments': len(flat_segments),
            'augmentation_factor': self.augmentation_factor,
            'splits': {name: len(segs) for name, segs in splits.items()},
            'source_distribution': {source: sources.count(source) for source in set(sources)},
            'segment_types': {
                'paragraph': len([s for s in flat_segments if 'paragraph' in s['type']]),
                'sliding_window': len([s for s in flat_segments if 'sliding_window' in s['type']]),
                'section': len([s for s in flat_segments if 'section' in s['type']]),
                'augmented': len([s for s in flat_segments if s.get('augmented', False)])
            },
            'token_stats': {
                'mean': np.mean([s['tokens'] for s in flat_segments]),
                'std': np.std([s['tokens'] for s in flat_segments]),
                'min': min([s['tokens'] for s in flat_segments]),
                'max': max([s['tokens'] for s in flat_segments])
            }
        }
        
        stats_file = self.output_dir / 'finetuning_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✅ Fine-tuning data ready! Statistics: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for Switch Transformer fine-tuning")
    parser.add_argument("--raw_pdf_dir", type=str, default="../data/raw_pdfs")
    parser.add_argument("--output_dir", type=str, default="../data")
    parser.add_argument("--tokenizer", type=str, default="google/switch-base-8")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--augmentation_factor", type=int, default=3)
    
    args = parser.parse_args()
    
    processor = FineTuningDataProcessor(
        raw_pdf_dir=args.raw_pdf_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length,
        augmentation_factor=args.augmentation_factor
    )
    
    all_segments = processor.process_all_pdfs()
    
    if all_segments:
        processor.create_finetuning_splits(all_segments)
        logger.info("✅ Fine-tuning data preparation completed!")
    else:
        logger.error("No data processed. Check your PDF files.")

if __name__ == "__main__":
    main()