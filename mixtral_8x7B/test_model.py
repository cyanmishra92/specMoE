#!/usr/bin/env python3
"""
Test script to debug model loading and trace collection
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    """Test model loading and simple inference"""
    
    # Load model
    logger.info("Loading DialoGPT-medium...")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/DialoGPT-medium",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        output_hidden_states=True,
        trust_remote_code=True,
    )
    
    logger.info("Model loaded successfully")
    
    # Test inference
    test_text = "Hello, this is a test sentence for the model."
    
    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    logger.info(f"Input shape: {inputs['input_ids'].shape}")
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True
        )
    
    logger.info("Forward pass successful")
    logger.info(f"Number of hidden states: {len(outputs.hidden_states)}")
    
    # Test synthetic routing
    hidden_states = outputs.hidden_states
    num_experts = 8
    
    for layer_idx in range(len(hidden_states) - 1):
        batch_size, seq_len, hidden_size = hidden_states[layer_idx].shape
        logger.info(f"Layer {layer_idx}: shape {(batch_size, seq_len, hidden_size)}")
        
        # Create synthetic routing
        synthetic_routing = torch.rand(batch_size, seq_len, num_experts)
        top_2_values, top_2_indices = torch.topk(synthetic_routing, k=2, dim=-1)
        
        logger.info(f"Layer {layer_idx}: synthetic routing shape {synthetic_routing.shape}")
        logger.info(f"Layer {layer_idx}: top-2 indices shape {top_2_indices.shape}")
        
        if layer_idx >= 2:  # Just test first few layers
            break
    
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    test_model()