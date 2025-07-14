#!/usr/bin/env python3
"""
Test script to verify diverse datasets work
"""

import datasets
import sys
from transformers import AutoTokenizer

def test_dataset_availability():
    """Test which datasets are available"""
    print("🧪 Testing dataset availability...")
    
    datasets_to_test = [
        {'name': 'cnn_dailymail', 'config': '3.0.0'},
        {'name': 'imdb', 'config': None},
        {'name': 'wikitext', 'config': 'wikitext-2-raw-v1'},
        {'name': 'squad', 'config': 'plain_text'},
        {'name': 'xsum', 'config': None},
        {'name': 'multi_news', 'config': None},
        {'name': 'billsum', 'config': None},
        {'name': 'reddit_tifu', 'config': 'short'},
        {'name': 'newsroom', 'config': None},
        {'name': 'scientific_papers', 'config': 'arxiv'},
        {'name': 'booksum', 'config': None},
        {'name': 'gigaword', 'config': None},
        {'name': 'samsum', 'config': None},
        {'name': 'dialogsum', 'config': None},
        {'name': 'amazon_reviews_multi', 'config': 'en'}
    ]
    
    available_datasets = []
    
    for dataset_config in datasets_to_test:
        try:
            print(f"Testing {dataset_config['name']}...")
            
            if dataset_config['config']:
                dataset = datasets.load_dataset(
                    dataset_config['name'], 
                    dataset_config['config'], 
                    split="train",
                    streaming=True
                )
            else:
                dataset = datasets.load_dataset(
                    dataset_config['name'], 
                    split="train",
                    streaming=True
                )
            
            # Test first sample
            sample = next(iter(dataset))
            print(f"✅ {dataset_config['name']}: Available")
            print(f"   Sample keys: {list(sample.keys())}")
            available_datasets.append(dataset_config)
            
        except Exception as e:
            print(f"❌ {dataset_config['name']}: {e}")
    
    print(f"\n📊 Results: {len(available_datasets)}/{len(datasets_to_test)} datasets available")
    
    return available_datasets

def test_small_collection():
    """Test collection with available datasets"""
    print(f"\n🚀 Testing small collection with real datasets...")
    
    import subprocess
    cmd = [
        sys.executable, 
        "scripts/collection/collect_robust_traces.py", 
        "--traces", "50",
        "--mode", "real"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, cwd="/data/research/specMoE/specMoE")
        if result.returncode == 0:
            print("✅ Small collection with real datasets successful!")
            return True
        else:
            print(f"❌ Collection failed with return code: {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ Collection failed with error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Diverse Dataset Collection")
    print("=" * 50)
    
    # Test dataset availability
    available_datasets = test_dataset_availability()
    
    if len(available_datasets) >= 4:
        print(f"\n✅ Sufficient datasets available ({len(available_datasets)})")
        
        # Test small collection
        success = test_small_collection()
        
        if success:
            print("\n🎉 Ready for large-scale collection!")
            print("Run: python scripts/collection/collect_robust_traces.py --traces 10000 --mode real")
        else:
            print("\n❌ Collection test failed")
    else:
        print(f"\n⚠️  Only {len(available_datasets)} datasets available, may need synthetic fallback")