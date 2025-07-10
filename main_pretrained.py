"""
Main script for Enhanced Pre-gated MoE using Pre-trained Switch Transformers
Supports both custom and pre-trained models
"""

import torch
import argparse
import json
from pathlib import Path
import sys

from models.pretrained_switch_model import create_pretrained_switch_model, get_available_switch_models
from models.small_switch_transformer import create_small_switch_model
from gating.speculation_engine import create_speculation_engine, SpeculativeGatingWrapper
from memory.adaptive_memory_manager import create_memory_manager
from utils.device_profiler import profile_current_device


def setup_pretrained_model_and_systems(args):
    """Set up pre-trained model and all supporting systems"""
    print("Setting up Enhanced Pre-gated MoE with Pre-trained Switch Transformer...")
    
    # Device profiling
    device_profile = profile_current_device()
    print(f"Device: {device_profile.device_name} ({device_profile.memory_capacity_gb:.1f} GB)")
    
    # Create pre-trained model
    try:
        model_wrapper = create_pretrained_switch_model(args.pretrained_model)
        model_info = model_wrapper.get_model_info()
        print(f"Model: {model_info['model_name']}")
        print(f"Total parameters: {model_info['total_parameters']:,}")
        print(f"Experts per layer: {model_info['num_experts_per_layer']}")
    except Exception as e:
        print(f"Failed to load pre-trained model: {e}")
        print("Falling back to custom small model...")
        model_wrapper = create_small_switch_model().cuda()
        model_info = model_wrapper.get_model_info()
    
    # Create speculation engine
    speculation_engine = create_speculation_engine(
        num_experts=model_info['num_experts_per_layer'],
        num_layers=model_info.get('num_layers', 6),
        mode=args.speculation_mode
    )
    print(f"Speculation mode: {args.speculation_mode}")
    
    # Create synthetic expert weights for memory manager (since we can't easily extract from pretrained)
    expert_weights = {}
    hidden_size = model_info['hidden_size']
    num_layers = model_info.get('num_layers', 6)
    
    for layer_id in range(num_layers):
        for expert_id in range(model_info['num_experts_per_layer']):
            # Create realistic expert weights based on model architecture
            expert_size = model_info.get('memory_per_expert_mb', 4) * 1024 * 1024 // 4  # Convert MB to float32 count
            expert_key = f"layer_{layer_id}_expert_{expert_id}"
            expert_weights[expert_key] = torch.randn(expert_size)
    
    # Create memory manager
    memory_manager = create_memory_manager(device_profile, model_wrapper, expert_weights)
    print(f"Memory strategy: {memory_manager.buffer_strategy.value}")
    print(f"Compression: {memory_manager.compression_type.value}")
    
    return model_wrapper, speculation_engine, memory_manager, device_profile, model_info


def run_pretrained_inference_demo(model_wrapper, args):
    """Run inference demo with pre-trained model"""
    print("\nRunning pre-trained model inference demonstration...")
    
    # Test texts for different capabilities
    test_texts = [
        "Translate to French: Hello, how are you today?",
        "Summarize: Artificial intelligence is transforming many industries including healthcare, finance, and transportation.",
        "Question: What is the capital of France? Answer:",
    ]
    
    print(f"Test inputs: {len(test_texts)} texts")
    
    # Prepare inputs
    inputs = model_wrapper.prepare_inputs(test_texts, max_length=256)
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model_wrapper.forward(inputs['input_ids'], inputs['attention_mask'])
        torch.cuda.synchronize()
    
    # Benchmark inference
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    with torch.no_grad():
        outputs = model_wrapper.forward(inputs['input_ids'], inputs['attention_mask'])
    end_time.record()
    torch.cuda.synchronize()
    
    inference_time = start_time.elapsed_time(end_time)
    batch_size, seq_len = inputs['input_ids'].shape
    tokens_per_second = (batch_size * seq_len) / (inference_time / 1000)
    
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Throughput: {tokens_per_second:.0f} tokens/sec")
    
    # Analyze routing information
    routing_info = outputs['routing_info']
    print(f"MoE layers with routing: {len(routing_info)}")
    
    if routing_info:
        # Analyze expert usage
        total_entropy = 0
        total_load_balance = 0
        expert_usage_stats = []
        
        for i, layer_info in enumerate(routing_info):
            entropy = layer_info['routing_entropy'].item()
            load_balance = layer_info['load_balancing_loss'].item()
            expert_usage = layer_info['expert_usage'].cpu().numpy()
            
            total_entropy += entropy
            total_load_balance += load_balance
            expert_usage_stats.append(expert_usage)
            
            print(f"Layer {i}: Entropy={entropy:.3f}, Load Balance={load_balance:.3f}")
        
        avg_entropy = total_entropy / len(routing_info)
        avg_load_balance = total_load_balance / len(routing_info)
        
        print(f"Average routing entropy: {avg_entropy:.3f}")
        print(f"Average load balance loss: {avg_load_balance:.3f}")
    
    return outputs


def run_pretrained_generation_demo(model_wrapper, args):
    """Run text generation demo with routing statistics"""
    print("\nRunning text generation with routing analysis...")
    
    # Generation prompts
    prompts = [
        "Translate to German: The weather is beautiful today.",
        "Summarize the benefits of renewable energy in one sentence."
    ]
    
    # Generate with routing stats
    generation_results = model_wrapper.generate_with_routing_stats(
        prompts,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=model_wrapper.tokenizer.eos_token_id
    )
    
    print("Generated texts:")
    for i, text in enumerate(generation_results['generated_texts']):
        print(f"{i+1}: {text}")
    
    return generation_results


def compare_models():
    """Compare custom vs pre-trained models"""
    print("\nComparing Custom vs Pre-trained Models...")
    
    device_profile = profile_current_device()
    results = {}
    
    # Test custom model
    print("\n--- Testing Custom Model ---")
    try:
        custom_model = create_small_switch_model().cuda()
        custom_info = custom_model.get_model_info()
        
        # Quick inference test
        input_ids = torch.randint(0, 32000, (2, 128)).cuda()
        
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.no_grad():
            custom_outputs = custom_model(input_ids)
        end_time.record()
        torch.cuda.synchronize()
        
        custom_time = start_time.elapsed_time(end_time)
        custom_throughput = (2 * 128) / (custom_time / 1000)
        
        results['custom'] = {
            'model_name': 'Custom Small Switch',
            'parameters': custom_info['total_parameters'],
            'inference_time_ms': custom_time,
            'tokens_per_second': custom_throughput,
            'experts_per_layer': custom_info['num_experts_per_layer'],
            'load_balancing_loss': custom_outputs['load_balancing_loss'].item()
        }
        
        print(f"✅ Custom model: {custom_time:.1f}ms, {custom_throughput:.0f} tokens/sec")
        
    except Exception as e:
        print(f"❌ Custom model failed: {e}")
        results['custom'] = {'error': str(e)}
    
    # Test pre-trained models
    available_models = get_available_switch_models()
    
    for model_info in available_models[:2]:  # Test first 2 models
        model_name = model_info['name']
        print(f"\n--- Testing {model_name} ---")
        
        try:
            pretrained_model = create_pretrained_switch_model(model_name)
            pretrained_info = pretrained_model.get_model_info()
            
            # Prepare test input
            test_text = ["Translate to French: Hello world"]
            inputs = pretrained_model.prepare_inputs(test_text, max_length=128)
            
            # Quick inference test
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            with torch.no_grad():
                pretrained_outputs = pretrained_model.forward(
                    inputs['input_ids'], 
                    inputs['attention_mask']
                )
            end_time.record()
            torch.cuda.synchronize()
            
            pretrained_time = start_time.elapsed_time(end_time)
            seq_len = inputs['input_ids'].shape[1]
            pretrained_throughput = seq_len / (pretrained_time / 1000)
            
            # Calculate average load balancing loss
            routing_info = pretrained_outputs['routing_info']
            avg_load_balance = sum(layer['load_balancing_loss'].item() for layer in routing_info) / len(routing_info)
            
            results[model_name] = {
                'model_name': model_name,
                'parameters': pretrained_info['total_parameters'],
                'inference_time_ms': pretrained_time,
                'tokens_per_second': pretrained_throughput,
                'experts_per_layer': pretrained_info['num_experts_per_layer'],
                'load_balancing_loss': avg_load_balance,
                'num_moe_layers': len(routing_info)
            }
            
            print(f"✅ {model_name}: {pretrained_time:.1f}ms, {pretrained_throughput:.0f} tokens/sec")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            results[model_name] = {'error': str(e)}
    
    # Print comparison
    print("\n" + "="*100)
    print("MODEL COMPARISON")
    print("="*100)
    print(f"{'Model':<30} {'Params':<15} {'Time (ms)':<12} {'Tokens/sec':<12} {'Experts':<10} {'Load Balance':<12}")
    print("-"*100)
    
    for model_name, data in results.items():
        if 'error' in data:
            print(f"{model_name:<30} {'ERROR':<15} {'-':<12} {'-':<12} {'-':<10} {'-':<12}")
        else:
            params = f"{data['parameters']/1e6:.1f}M"
            print(f"{data['model_name']:<30} {params:<15} {data['inference_time_ms']:<12.1f} {data['tokens_per_second']:<12.0f} {data['experts_per_layer']:<10} {data['load_balancing_loss']:<12.3f}")
    
    # Save results
    with open('model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Enhanced Pre-gated MoE with Pre-trained Models")
    parser.add_argument('--mode', choices=['demo', 'generation', 'compare', 'list-models'], default='demo',
                       help='Run mode: demo, generation, compare, or list-models')
    
    # Model selection
    parser.add_argument('--pretrained-model', default='google/switch-base-8',
                       help='Pre-trained model name (default: google/switch-base-8)')
    parser.add_argument('--use-custom', action='store_true',
                       help='Use custom small model instead of pre-trained')
    
    # Speculation parameters
    parser.add_argument('--speculation-mode', default='multi_layer',
                       choices=['none', 'layer_minus_1', 'multi_layer', 'pattern', 'adaptive'],
                       help='Speculation mode to use')
    
    args = parser.parse_args()
    
    print("Enhanced Pre-gated MoE for RTX 3090 - Pre-trained Models")
    print("=" * 60)
    
    if args.mode == 'list-models':
        print("Available Switch Transformer models:")
        for model in get_available_switch_models():
            print(f"  📦 {model['name']}")
            print(f"     {model['description']}")
            print(f"     Size: {model['size']}, Experts: {model['experts']}")
            print()
        return
    
    elif args.mode == 'compare':
        comparison_results = compare_models()
        print(f"\nComparison results saved to 'model_comparison.json'")
        return
    
    # Setup model and systems
    if args.use_custom:
        print("Using custom small model...")
        # Use original main.py logic
        from main import setup_model_and_systems, run_inference_demo
        enhanced_model, memory_manager, device_profile, model_info = setup_model_and_systems(args)
        outputs = run_inference_demo(enhanced_model, args)
    else:
        model_wrapper, speculation_engine, memory_manager, device_profile, model_info = setup_pretrained_model_and_systems(args)
        
        if args.mode == 'demo':
            outputs = run_pretrained_inference_demo(model_wrapper, args)
        elif args.mode == 'generation':
            outputs = run_pretrained_generation_demo(model_wrapper, args)
    
    # Show memory statistics
    memory_stats = memory_manager.get_memory_stats()
    print(f"\nMemory Statistics:")
    print(f"GPU cache hit rate: {memory_stats['gpu_cache']['hit_rate']:.3f}")
    print(f"Average load time: {memory_stats['avg_load_time_ms']:.2f} ms")
    if memory_stats.get('compression_ratio', 1.0) > 1.0:
        print(f"Compression ratio: {memory_stats['compression_ratio']:.1f}x")
    
    print("\nDone!")


if __name__ == "__main__":
    main()