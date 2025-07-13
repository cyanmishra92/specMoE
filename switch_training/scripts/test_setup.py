#!/usr/bin/env python3
"""
Test Setup Script

Verify that all components are working before starting training.
"""

import os
import sys
import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration, T5ForConditionalGeneration

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cuda_setup():
    """Test CUDA availability and memory"""
    logger.info("🔧 Testing CUDA setup...")
    
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ CUDA available: {device}")
        logger.info(f"✅ GPU memory: {memory:.1f} GB")
        
        # Test memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            logger.info("✅ GPU memory allocation test passed")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"❌ GPU memory test failed: {e}")
            return False
    else:
        logger.warning("⚠️ CUDA not available, will use CPU")
    
    return True

def test_data_loading():
    """Test data loading"""
    logger.info("📊 Testing data loading...")
    
    data_dir = Path("../data")
    
    # Check if data files exist
    files_to_check = [
        "train/finetuning_train.json",
        "val/finetuning_val.json", 
        "test/finetuning_test.json",
        "finetuning_stats.json"
    ]
    
    for file_path in files_to_check:
        full_path = data_dir / file_path
        if full_path.exists():
            logger.info(f"✅ Found: {file_path}")
            
            # Test loading
            try:
                with open(full_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"   - Loaded {len(data)} items" if isinstance(data, list) else "   - Config file loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load {file_path}: {e}")
                return False
        else:
            logger.error(f"❌ Missing: {file_path}")
            return False
    
    return True

def test_model_loading():
    """Test model and tokenizer loading"""
    logger.info("🤖 Testing model loading...")
    
    # Test models to try (in order of preference for RTX 3090)
    models_to_test = [
        "google/switch-base-16",   # Optimal balance
        "google/switch-base-8",    # Most conservative
        "google/switch-base-32",   # More ambitious
        "t5-small"                 # Fallback
    ]
    
    for model_name in models_to_test:
        try:
            logger.info(f"Testing {model_name}...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"✅ Tokenizer loaded: {len(tokenizer)} vocab size")
            
            # Load model (small test) - use appropriate class
            if "switch" in model_name.lower():
                model = SwitchTransformersForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            elif "t5" in model_name.lower():
                model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            else:
                continue  # Skip unsupported models
            
            # Test forward pass (seq2seq style)
            test_input = "Summarize: This is a test sentence for the model."
            test_target = "This is a test."
            
            inputs = tokenizer(test_input, return_tensors="pt", max_length=50, truncation=True)
            labels = tokenizer(test_target, return_tensors="pt", max_length=50, truncation=True)['input_ids']
            
            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
            
            logger.info(f"✅ Model loaded and tested: {model_name}")
            logger.info(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
            logger.info(f"   - Test loss: {loss.item():.4f}")
            
            # Clean up
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {model_name}: {e}")
            continue
    
    logger.error("❌ No models could be loaded")
    return False

def test_training_sample():
    """Test a minimal training sample"""
    logger.info("🏋️ Testing training sample...")
    
    try:
        # Load small amount of data
        data_file = Path("../data/train/finetuning_train.json")
        with open(data_file, 'r') as f:
            train_data = json.load(f)[:5]  # Just 5 samples
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test tokenization
        for i, item in enumerate(train_data):
            text = item['text']
            tokens = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            logger.info(f"Sample {i+1}: {len(text)} chars → {tokens['input_ids'].size(1)} tokens")
        
        logger.info("✅ Training data processing test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training sample test failed: {e}")
        return False

def test_directories():
    """Test directory structure"""
    logger.info("📁 Testing directory structure...")
    
    dirs_to_check = [
        "../data/train",
        "../data/val", 
        "../data/test",
        "../data/processed",
        "../models",
        "../logs",
        "../benchmarks"
    ]
    
    for dir_path in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            logger.info(f"✅ Directory exists: {dir_path}")
        else:
            logger.info(f"📁 Creating directory: {dir_path}")
            path.mkdir(parents=True, exist_ok=True)
    
    return True

def main():
    """Run all tests"""
    logger.info("🧪 SWITCH TRANSFORMER SETUP TEST")
    logger.info("=" * 50)
    
    tests = [
        ("CUDA Setup", test_cuda_setup),
        ("Directory Structure", test_directories), 
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model_loading),
        ("Training Sample", test_training_sample)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\\n🔍 Running {test_name} test...")
        try:
            if test_func():
                logger.info(f"✅ {test_name} - PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} - FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERROR: {e}")
    
    logger.info(f"\\n📊 TEST SUMMARY")
    logger.info(f"=" * 30)
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Ready to start training.")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)