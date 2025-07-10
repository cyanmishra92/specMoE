#!/usr/bin/env python3
"""
Debug Switch Transformer Router Logits
Quick test to see what's actually returned by the model
"""

import torch
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_switch_outputs():
    """Debug what Switch Transformer actually returns"""
    
    logger.info("🔍 Debugging Switch Transformer Router Logits")
    
    try:
        # Load model and tokenizer
        logger.info("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")  # Use smaller model for faster debug
        model = SwitchTransformersForConditionalGeneration.from_pretrained(
            "google/switch-base-8",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        logger.info(f"Model device: {next(model.parameters()).device}")
        
        # Test input
        text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        logger.info(f"Input shape: {inputs['input_ids'].shape}")
        logger.info(f"Input text: {text}")
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Test different API calls
        logger.info("\n🧪 Testing different API calls:")
        
        # Method 1: Standard forward
        logger.info("1. Standard forward pass...")
        with torch.no_grad():
            outputs1 = model(**inputs)
        
        logger.info(f"Standard outputs type: {type(outputs1)}")
        logger.info(f"Standard outputs attributes: {dir(outputs1)}")
        
        # Method 2: With output_router_logits=True
        logger.info("\n2. With output_router_logits=True...")
        with torch.no_grad():
            outputs2 = model(**inputs, output_router_logits=True)
        
        logger.info(f"Router outputs type: {type(outputs2)}")
        logger.info(f"Router outputs attributes: {[attr for attr in dir(outputs2) if not attr.startswith('_')]}")
        
        # Check for router logits
        if hasattr(outputs2, 'encoder_router_logits'):
            logger.info(f"✅ Found encoder_router_logits!")
            logger.info(f"Type: {type(outputs2.encoder_router_logits)}")
            if outputs2.encoder_router_logits is not None:
                if isinstance(outputs2.encoder_router_logits, (list, tuple)):
                    logger.info(f"Length: {len(outputs2.encoder_router_logits)}")
                    for i, logits in enumerate(outputs2.encoder_router_logits):
                        if logits is not None:
                            logger.info(f"Layer {i} logits shape: {logits.shape}")
                        else:
                            logger.info(f"Layer {i} logits: None")
                else:
                    logger.info(f"Shape: {outputs2.encoder_router_logits.shape}")
            else:
                logger.warning("encoder_router_logits is None")
        else:
            logger.warning("❌ No encoder_router_logits found")
        
        if hasattr(outputs2, 'decoder_router_logits'):
            logger.info(f"✅ Found decoder_router_logits!")
            logger.info(f"Type: {type(outputs2.decoder_router_logits)}")
            if outputs2.decoder_router_logits is not None:
                if isinstance(outputs2.decoder_router_logits, (list, tuple)):
                    logger.info(f"Length: {len(outputs2.decoder_router_logits)}")
                else:
                    logger.info(f"Shape: {outputs2.decoder_router_logits.shape}")
            else:
                logger.warning("decoder_router_logits is None")
        else:
            logger.warning("❌ No decoder_router_logits found")
        
        # Method 3: With labels (conditional generation)
        logger.info("\n3. With labels (conditional generation)...")
        labels = inputs['input_ids'].clone()
        with torch.no_grad():
            outputs3 = model(**inputs, labels=labels, output_router_logits=True)
        
        logger.info(f"Conditional outputs type: {type(outputs3)}")
        if hasattr(outputs3, 'encoder_router_logits'):
            logger.info(f"✅ Conditional encoder_router_logits found!")
            if outputs3.encoder_router_logits is not None:
                if isinstance(outputs3.encoder_router_logits, (list, tuple)):
                    logger.info(f"Length: {len(outputs3.encoder_router_logits)}")
                    for i, logits in enumerate(outputs3.encoder_router_logits):
                        if logits is not None:
                            logger.info(f"Layer {i} logits shape: {logits.shape}")
                            # Show actual values
                            logger.info(f"Layer {i} logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
                            # Check expert distribution
                            gate_scores = torch.softmax(logits, dim=-1)
                            top_experts = torch.argmax(gate_scores, dim=-1)
                            unique_experts = torch.unique(top_experts)
                            logger.info(f"Layer {i} unique experts used: {unique_experts.tolist()}")
                else:
                    logger.info(f"Shape: {outputs3.encoder_router_logits.shape}")
            else:
                logger.warning("Conditional encoder_router_logits is None")
        
        # Method 4: Check model config
        logger.info("\n4. Model configuration...")
        logger.info(f"Model config: {model.config}")
        if hasattr(model.config, 'num_experts'):
            logger.info(f"Config num_experts: {model.config.num_experts}")
        if hasattr(model.config, 'expert_capacity'):
            logger.info(f"Config expert_capacity: {model.config.expert_capacity}")
        
        # Method 5: Check model architecture
        logger.info("\n5. Model architecture inspection...")
        moe_layers = []
        for name, module in model.named_modules():
            if 'switch' in name.lower() or 'expert' in name.lower() or 'router' in name.lower():
                moe_layers.append(name)
        
        logger.info(f"MoE-related layers found: {len(moe_layers)}")
        for layer in moe_layers[:5]:  # Show first 5
            logger.info(f"  {layer}")
        if len(moe_layers) > 5:
            logger.info(f"  ... and {len(moe_layers) - 5} more")
        
        return True
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_switch_outputs()
    if success:
        print("\n✅ Debug completed!")
    else:
        print("\n❌ Debug failed!")