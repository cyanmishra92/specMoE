#!/usr/bin/env python3
"""
Final Debug for Switch Transformer Router Logits
Handle nested tuple structure properly
"""

import torch
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_router_structure():
    """Debug the exact structure of router logits"""
    
    logger.info("🔍 Final Router Logits Debug")
    
    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
        model = SwitchTransformersForConditionalGeneration.from_pretrained(
            "google/switch-base-8",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        # Simple input
        input_text = "translate English to German: Hello world"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                decoder_input_ids=inputs['input_ids'],  # Use same for simplicity
                output_router_logits=True
            )
        
        logger.info("📊 Router Logits Structure Analysis:")
        
        # Analyze encoder router logits
        if hasattr(outputs, 'encoder_router_logits') and outputs.encoder_router_logits is not None:
            encoder_logits = outputs.encoder_router_logits
            logger.info(f"Encoder router logits type: {type(encoder_logits)}")
            logger.info(f"Encoder router logits length: {len(encoder_logits)}")
            
            all_experts_used = set()
            
            for i, layer_data in enumerate(encoder_logits):
                logger.info(f"\nLayer {i}:")
                logger.info(f"  Type: {type(layer_data)}")
                
                if layer_data is None:
                    logger.info(f"  Layer {i}: None (no MoE)")
                    continue
                
                if isinstance(layer_data, tuple):
                    logger.info(f"  Tuple length: {len(layer_data)}")
                    for j, item in enumerate(layer_data):
                        if item is not None and hasattr(item, 'shape'):
                            logger.info(f"    Item {j}: {item.shape}")
                            # Try to process as router logits
                            if len(item.shape) == 3:  # [batch, seq, experts]
                                gate_scores = torch.softmax(item, dim=-1)
                                top_experts = torch.argmax(gate_scores, dim=-1)
                                unique_experts = torch.unique(top_experts).cpu().tolist()
                                all_experts_used.update(unique_experts)
                                logger.info(f"      Experts used: {unique_experts}")
                        else:
                            logger.info(f"    Item {j}: {type(item)} - {item}")
                
                elif hasattr(layer_data, 'shape'):
                    logger.info(f"  Tensor shape: {layer_data.shape}")
                    if len(layer_data.shape) == 3:  # [batch, seq, experts]
                        gate_scores = torch.softmax(layer_data, dim=-1)
                        top_experts = torch.argmax(gate_scores, dim=-1)
                        unique_experts = torch.unique(top_experts).cpu().tolist()
                        all_experts_used.update(unique_experts)
                        logger.info(f"    Experts used: {unique_experts}")
                
                else:
                    logger.info(f"  Unknown structure: {layer_data}")
            
            logger.info(f"\n🎯 Total experts used across all encoder layers: {sorted(all_experts_used)}")
            logger.info(f"🎯 Expert diversity: {len(all_experts_used)}/8 ({len(all_experts_used)/8*100:.1f}%)")
        
        # Analyze decoder router logits  
        if hasattr(outputs, 'decoder_router_logits') and outputs.decoder_router_logits is not None:
            decoder_logits = outputs.decoder_router_logits
            logger.info(f"\nDecoder router logits type: {type(decoder_logits)}")
            logger.info(f"Decoder router logits length: {len(decoder_logits)}")
            
            decoder_experts_used = set()
            
            for i, layer_data in enumerate(decoder_logits):
                if layer_data is None:
                    continue
                
                if isinstance(layer_data, tuple):
                    for j, item in enumerate(layer_data):
                        if item is not None and hasattr(item, 'shape') and len(item.shape) == 3:
                            gate_scores = torch.softmax(item, dim=-1)
                            top_experts = torch.argmax(gate_scores, dim=-1)
                            unique_experts = torch.unique(top_experts).cpu().tolist()
                            decoder_experts_used.update(unique_experts)
                
                elif hasattr(layer_data, 'shape') and len(layer_data.shape) == 3:
                    gate_scores = torch.softmax(layer_data, dim=-1)
                    top_experts = torch.argmax(gate_scores, dim=-1)
                    unique_experts = torch.unique(top_experts).cpu().tolist()
                    decoder_experts_used.update(unique_experts)
            
            logger.info(f"\n🎯 Total experts used in decoder: {sorted(decoder_experts_used)}")
            logger.info(f"🎯 Decoder diversity: {len(decoder_experts_used)}/8 ({len(decoder_experts_used)/8*100:.1f}%)")
            
            # Combined diversity
            combined_experts = all_experts_used.union(decoder_experts_used)
            logger.info(f"\n🌟 Combined encoder+decoder experts: {sorted(combined_experts)}")
            logger.info(f"🌟 Total diversity: {len(combined_experts)}/8 ({len(combined_experts)/8*100:.1f}%)")
        
        return True, len(all_experts_used), encoder_logits
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, None

def create_simple_trace_example(encoder_logits):
    """Create a simple trace extraction example"""
    
    logger.info("\n🧪 Creating trace extraction example...")
    
    try:
        if encoder_logits is None:
            logger.warning("No encoder logits to process")
            return
        
        traces_created = 0
        
        for layer_idx, layer_data in enumerate(encoder_logits):
            if layer_data is None:
                continue
            
            # Handle different data structures
            router_logits = None
            
            if isinstance(layer_data, tuple):
                # Find the router logits tensor in the tuple
                for item in layer_data:
                    if item is not None and hasattr(item, 'shape') and len(item.shape) == 3:
                        router_logits = item
                        break
            elif hasattr(layer_data, 'shape') and len(layer_data.shape) == 3:
                router_logits = layer_data
            
            if router_logits is not None:
                logger.info(f"Layer {layer_idx}: Found router logits {router_logits.shape}")
                
                # Convert to gate scores
                gate_scores = torch.softmax(router_logits, dim=-1)
                top_experts = torch.topk(gate_scores, k=1, dim=-1)
                
                logger.info(f"  Gate scores shape: {gate_scores.shape}")
                logger.info(f"  Top experts shape: {top_experts.indices.shape}")
                
                # Show expert distribution
                expert_usage = {}
                flat_experts = top_experts.indices.flatten().cpu().tolist()
                for expert in flat_experts:
                    expert_usage[expert] = expert_usage.get(expert, 0) + 1
                
                logger.info(f"  Expert usage: {expert_usage}")
                traces_created += 1
        
        logger.info(f"✅ Successfully processed {traces_created} layers with router logits")
        return traces_created
        
    except Exception as e:
        logger.error(f"Trace creation failed: {e}")
        return 0

if __name__ == "__main__":
    success, expert_count, encoder_logits = debug_router_structure()
    if success:
        print(f"\n✅ Router structure analyzed! Found {expert_count} unique experts")
        trace_count = create_simple_trace_example(encoder_logits)
        print(f"✅ Created {trace_count} example traces")
    else:
        print("\n❌ Debug failed!")