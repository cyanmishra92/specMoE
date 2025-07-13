#!/usr/bin/env python3
"""
Debug data preprocessing issues
"""

import json
from pathlib import Path

def check_data_issues():
    """Check for data issues that might cause NaN"""
    
    # Load sample data
    data_file = Path("../data/train/finetuning_train.json")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    problematic_items = []
    
    for i, item in enumerate(data[:100]):  # Check first 100
        text = item['text']
        
        # Check for issues
        issues = []
        
        if not text or text.strip() == "":
            issues.append("empty_text")
        
        if len(text) < 10:
            issues.append("too_short")
        
        # Try splitting for seq2seq
        words = text.split()
        if len(words) < 4:
            issues.append("insufficient_words")
        else:
            split_point = int(len(words) * 0.7)
            source_words = words[:split_point]
            target_words = words[split_point:]
            
            source_text = "continue: " + " ".join(source_words)
            target_text = " ".join(target_words)
            
            if not target_text.strip():
                issues.append("empty_target")
            
            if len(target_text.strip()) < 3:
                issues.append("target_too_short")
            
            # Check for problematic characters
            try:
                target_text.encode('utf-8')
            except:
                issues.append("encoding_issue")
        
        if issues:
            problematic_items.append({
                'index': i,
                'text': text[:100] + "..." if len(text) > 100 else text,
                'issues': issues,
                'length': len(text),
                'words': len(words) if 'words' in locals() else 0
            })
    
    print(f"\nFound {len(problematic_items)} problematic items:")
    for item in problematic_items[:10]:  # Show first 10
        print(f"Index {item['index']}: {item['issues']}")
        print(f"  Text: {item['text']}")
        print(f"  Length: {item['length']}, Words: {item['words']}\n")
    
    return problematic_items

if __name__ == "__main__":
    issues = check_data_issues()