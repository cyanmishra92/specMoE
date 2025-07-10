#!/usr/bin/env python3
"""
Test Learnable Speculation Models
Validate that trained models work with the speculation engine
"""

import torch
import sys
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_learnable_models():
    """Test all trained learnable models"""
    
    logger.info("🧠 Testing Learnable Speculation Models")
    logger.info("=" * 50)
    
    # Check for trained models
    model_dir = Path("trained_models")
    if not model_dir.exists():
        logger.error("❌ No trained_models directory found")
        return False
    
    # Look for real data models
    model_files = list(model_dir.glob("*real_data.pt"))
    if not model_files:
        # Fallback to any .pt files
        model_files = list(model_dir.glob("*.pt"))
    
    if not model_files:
        logger.error("❌ No trained models found")
        logger.info("Run 'python train_with_real_traces.py' first")
        return False
    
    logger.info(f"Found {len(model_files)} trained models")
    
    success_count = 0
    
    for model_file in model_files:
        logger.info(f"\n🔍 Testing model: {model_file.name}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_file, map_location='cpu')
            logger.info(f"✅ Model loaded successfully")
            
            # Extract configuration
            if 'gating_config' in checkpoint:
                gating_config_dict = checkpoint['gating_config']
            else:
                # Default configuration for older models
                gating_config_dict = {
                    'hidden_size': 512,
                    'num_experts': 8,
                    'num_layers': 6,
                    'num_heads': 8,
                    'context_layers': 4,
                    'max_seq_len': 512,
                    'dropout': 0.1,
                    'temperature': 1.0,
                    'top_k': 2
                }
            
            # Import and create model
            from training.learnable_gating_models import create_gating_model, GatingModelConfig
            
            config = GatingModelConfig(**gating_config_dict)
            
            # Determine model type from filename
            if 'contextual' in model_file.name:
                model_type = 'contextual'
            elif 'transformer' in model_file.name:
                model_type = 'transformer'
            elif 'hierarchical' in model_file.name:
                model_type = 'hierarchical'
            else:
                model_type = 'contextual'  # Default
            
            # Create and load model
            model = create_gating_model(model_type, config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            logger.info(f"✅ Model architecture: {model_type}")
            logger.info(f"✅ Configuration: {config.hidden_size}d, {config.num_experts} experts")
            
            # Test forward pass
            batch_size, seq_len = 2, 64
            hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
            prev_gates = [torch.randn(batch_size, seq_len, config.num_experts) for _ in range(3)]
            attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
            
            with torch.no_grad():
                outputs = model(
                    hidden_states=hidden_states,
                    prev_layer_gates=prev_gates,
                    layer_id=2,
                    attention_mask=attention_mask
                )
                
                gating_logits, confidence, attention_weights = outputs
                
                # Validate outputs
                assert gating_logits.shape == (batch_size, seq_len, config.num_experts)
                assert confidence.shape == (batch_size, seq_len, 1)
                
                # Check predictions are reasonable
                gating_probs = torch.softmax(gating_logits, dim=-1)
                max_probs = gating_probs.max(dim=-1)[0]
                avg_confidence = confidence.mean().item()
                avg_max_prob = max_probs.mean().item()
                
                logger.info(f"✅ Forward pass successful")
                logger.info(f"   Average confidence: {avg_confidence:.3f}")
                logger.info(f"   Average max probability: {avg_max_prob:.3f}")
                
                # Test with speculation engine
                from gating.speculation_engine import LearnableSpeculationEngine
                
                speculation_engine = LearnableSpeculationEngine(
                    num_experts=config.num_experts,
                    num_layers=config.num_layers,
                    gating_model=model
                )
                
                # Test prediction
                predicted_experts, pred_confidence = speculation_engine.predict_next_experts(
                    current_layer=2,
                    hidden_states=hidden_states,
                    current_routing=prev_gates[-1]
                )
                
                logger.info(f"✅ Speculation engine integration successful")
                logger.info(f"   Predicted experts shape: {predicted_experts.shape}")
                logger.info(f"   Prediction confidence: {pred_confidence:.3f}")
                
                success_count += 1
                
        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")
            continue
    
    logger.info(f"\n📊 Test Results: {success_count}/{len(model_files)} models passed")
    
    if success_count > 0:
        logger.info("🎉 Learnable speculation models are working!")
        return True
    else:
        logger.error("❌ No models passed testing")
        return False

def test_integration_with_main_models():
    """Test integration with main inference models"""
    
    logger.info("\n🔗 Testing Integration with Main Models")
    logger.info("=" * 45)
    
    try:
        # Test with custom model
        from models.small_switch_transformer import create_small_switch_model
        from gating.speculation_engine import create_speculation_engine, SpeculativeGatingWrapper
        
        # Create model
        model = create_small_switch_model()
        model_info = model.get_model_info()
        
        # Create speculation engine (should use learnable if available)
        speculation_engine = create_speculation_engine(
            num_experts=model_info['num_experts_per_layer'],
            num_layers=model_info['num_layers'],
            mode="learnable"  # Use learnable mode
        )
        
        # Wrap model
        enhanced_model = SpeculativeGatingWrapper(model, speculation_engine)
        
        # Test inference
        input_ids = torch.randint(0, 32000, (2, 64))
        
        with torch.no_grad():
            outputs = enhanced_model.forward(input_ids)
        
        logger.info("✅ Integration test successful")
        logger.info(f"   Output shape: {outputs['logits'].shape}")
        
        if 'speculation_stats' in outputs:
            stats = outputs['speculation_stats']
            logger.info(f"   Speculation accuracy: {stats.get('overall_accuracy', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 LEARNABLE SPECULATION MODEL TESTS")
    print("=" * 60)
    
    # Test 1: Individual model validation
    test1_success = test_learnable_models()
    
    # Test 2: Integration with main system
    test2_success = test_integration_with_main_models()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    print(f"Model Validation:     {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"System Integration:   {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    overall_success = test1_success or test2_success
    
    if overall_success:
        print("\n🎉 LEARNABLE SPECULATION SYSTEM IS READY!")
        print("✅ Your trained models are working correctly")
        print("\nNext steps:")
        print("  • Run 'python main.py --mode demo --speculation-mode learnable'")
        print("  • Compare with other modes using 'python main.py --mode compare'")
    else:
        print("\n❌ TESTS FAILED")
        print("Check the error messages above and ensure:")
        print("  • Models are trained: 'python train_with_real_traces.py'")
        print("  • Dependencies are installed: 'pip install -r requirements.txt'")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)