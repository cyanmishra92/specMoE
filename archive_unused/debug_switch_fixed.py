#!/usr/bin/env python3
"""
Fixed Debug for Switch Transformer Router Logits
Switch Transformers are seq2seq models - need decoder_input_ids!
"""

import torch
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_switch_seq2seq():
    """Debug Switch Transformer with correct seq2seq setup"""
    
    logger.info("🔍 Debugging Switch Transformer (Seq2Seq)")
    
    try:
        # Load model and tokenizer
        logger.info("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
        model = SwitchTransformersForConditionalGeneration.from_pretrained(
            "google/switch-base-8",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        # Test seq2seq input/output
        input_text = "translate English to German: The quick brown fox"
        target_text = "Der schnelle braune Fuchs"
        
        # Tokenize input and target
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        targets = tokenizer(target_text, return_tensors="pt", padding=True, truncation=True)
        
        logger.info(f"Input: {input_text}")
        logger.info(f"Target: {target_text}")
        logger.info(f"Input shape: {inputs['input_ids'].shape}")
        logger.info(f"Target shape: {targets['input_ids'].shape}")
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # Method 1: Forward pass with decoder_input_ids
        logger.info("\n🧪 Method 1: With decoder_input_ids")
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=targets['input_ids'],
                output_router_logits=True
            )
        
        logger.info(f"Outputs type: {type(outputs)}")
        logger.info(f"Outputs keys: {[k for k in outputs.keys() if not k.startswith('_')]}")
        
        # Check encoder router logits
        if hasattr(outputs, 'encoder_router_logits') and outputs.encoder_router_logits is not None:
            logger.info("✅ Found encoder_router_logits!")
            encoder_logits = outputs.encoder_router_logits
            logger.info(f"Type: {type(encoder_logits)}")
            
            if isinstance(encoder_logits, (list, tuple)):
                logger.info(f"Number of encoder layers: {len(encoder_logits)}")
                for i, logits in enumerate(encoder_logits):
                    if logits is not None:
                        logger.info(f"  Encoder Layer {i}: {logits.shape}")
                        # Analyze expert usage
                        gate_scores = torch.softmax(logits, dim=-1)
                        top_experts = torch.argmax(gate_scores, dim=-1)
                        unique_experts = torch.unique(top_experts).cpu().tolist()
                        logger.info(f"    Experts used: {unique_experts} ({len(unique_experts)}/8 experts)")
                    else:
                        logger.info(f"  Encoder Layer {i}: None")
            else:
                logger.info(f"Single tensor shape: {encoder_logits.shape}")
        else:
            logger.warning("❌ No encoder_router_logits found")
        
        # Check decoder router logits
        if hasattr(outputs, 'decoder_router_logits') and outputs.decoder_router_logits is not None:
            logger.info("✅ Found decoder_router_logits!")
            decoder_logits = outputs.decoder_router_logits
            logger.info(f"Type: {type(decoder_logits)}")
            
            if isinstance(decoder_logits, (list, tuple)):
                logger.info(f"Number of decoder layers: {len(decoder_logits)}")
                for i, logits in enumerate(decoder_logits):
                    if logits is not None:
                        logger.info(f"  Decoder Layer {i}: {logits.shape}")
                        gate_scores = torch.softmax(logits, dim=-1)
                        top_experts = torch.argmax(gate_scores, dim=-1)
                        unique_experts = torch.unique(top_experts).cpu().tolist()
                        logger.info(f"    Experts used: {unique_experts} ({len(unique_experts)}/8 experts)")
                    else:
                        logger.info(f"  Decoder Layer {i}: None")
            else:
                logger.info(f"Single tensor shape: {decoder_logits.shape}")
        else:
            logger.warning("❌ No decoder_router_logits found")
        
        # Method 2: Training mode with labels
        logger.info("\n🧪 Method 2: Training mode with labels")
        with torch.no_grad():
            outputs2 = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=targets['input_ids'],  # This creates decoder_input_ids automatically
                output_router_logits=True
            )
        
        logger.info(f"Training outputs keys: {[k for k in outputs2.keys() if not k.startswith('_')]}")
        
        if hasattr(outputs2, 'encoder_router_logits') and outputs2.encoder_router_logits is not None:
            logger.info("✅ Training mode: encoder_router_logits found!")
            encoder_logits = outputs2.encoder_router_logits
            if isinstance(encoder_logits, (list, tuple)):
                total_experts_used = set()
                for i, logits in enumerate(encoder_logits):
                    if logits is not None:
                        gate_scores = torch.softmax(logits, dim=-1)
                        top_experts = torch.argmax(gate_scores, dim=-1)
                        unique_experts = torch.unique(top_experts).cpu().tolist()
                        total_experts_used.update(unique_experts)
                        logger.info(f"  Training Layer {i}: {len(unique_experts)} experts")
                
                logger.info(f"🎯 Total unique experts across all layers: {sorted(total_experts_used)}")
                logger.info(f"🎯 Diversity: {len(total_experts_used)}/8 experts used ({len(total_experts_used)/8*100:.1f}%)")
        
        # Test with different texts for more diversity
        logger.info("\n🧪 Method 3: Testing diverse texts")
        test_texts = [
            "translate English to French: Hello world",
            "summarize: The quick brown fox jumps over the lazy dog in the forest",
            "question: What is the capital of France? context: France is in Europe",
            "sentiment: This movie is absolutely amazing and wonderful"
        ]
        
        all_experts_used = set()
        for text in test_texts:
            inputs_test = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs_test = {k: v.to(device) for k, v in inputs_test.items()}
            
            with torch.no_grad():
                outputs_test = model(
                    input_ids=inputs_test['input_ids'],
                    decoder_input_ids=inputs_test['input_ids'],  # Self-input for simplicity
                    output_router_logits=True
                )
            
            if hasattr(outputs_test, 'encoder_router_logits') and outputs_test.encoder_router_logits is not None:
                for layer_logits in outputs_test.encoder_router_logits:
                    if layer_logits is not None:
                        gate_scores = torch.softmax(layer_logits, dim=-1)
                        top_experts = torch.argmax(gate_scores, dim=-1)
                        experts = torch.unique(top_experts).cpu().tolist()
                        all_experts_used.update(experts)
        
        logger.info(f"🌟 Across diverse texts: {sorted(all_experts_used)} ({len(all_experts_used)}/8 experts)")
        
        return True, len(all_experts_used)
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

if __name__ == "__main__":
    success, expert_count = debug_switch_seq2seq()
    if success:
        print(f"\n✅ Debug completed! Found {expert_count}/8 experts in use")
    else:
        print("\n❌ Debug failed!")