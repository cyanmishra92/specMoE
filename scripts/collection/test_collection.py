#!/usr/bin/env python3
"""
Test script for trace collection with small number of traces
"""

import subprocess
import sys
import os

def test_collection():
    """Test trace collection with small number"""
    print("🧪 Testing trace collection with small dataset...")
    
    # Test with small number of traces
    cmd = [
        sys.executable, 
        "scripts/collection/collect_robust_traces.py", 
        "--traces", "100",
        "--mode", "mixed",
        "--real-ratio", "0.3"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, cwd="/data/research/specMoE/specMoE")
        if result.returncode == 0:
            print("✅ Test collection successful!")
            return True
        else:
            print(f"❌ Test collection failed with return code: {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ Test collection failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_collection()
    sys.exit(0 if success else 1)