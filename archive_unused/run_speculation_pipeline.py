#!/usr/bin/env python3
"""
Unified Speculation Pipeline for Enhanced Pre-gated MoE
Complete workflow: Data Collection → Training → Testing → Evaluation
"""

import sys
import os
import subprocess
import time
import logging
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a shell command and handle errors"""
    logger.info(f"🚀 {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False

def check_prerequisites():
    """Check if required dependencies are available"""
    logger.info("🔍 Checking prerequisites...")
    
    # Check Python packages
    required_packages = ['torch', 'transformers', 'datasets', 'numpy', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {missing_packages}")
        logger.info("Install with: pip install " + " ".join(missing_packages))
        return False
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning("⚠️  No GPU detected - will use CPU (slower)")
    except Exception as e:
        logger.error(f"❌ Error checking GPU: {e}")
        return False
    
    logger.info("✅ All prerequisites met")
    return True

def step1_collect_traces(args):
    """Step 1: Collect routing traces from real MoE models"""
    logger.info("=" * 60)
    logger.info("STEP 1: COLLECTING ROUTING TRACES")
    logger.info("=" * 60)
    
    # Check if traces already exist
    trace_file = Path("routing_data/proper_traces.pkl")
    if trace_file.exists() and not args.force_recollect:
        size_mb = trace_file.stat().st_size / (1024*1024)
        logger.info(f"✅ Traces already exist: {trace_file} ({size_mb:.1f} MB)")
        logger.info("Use --force-recollect to regenerate traces")
        return True
    
    # Run trace collection
    success = run_command(
        "python collect_real_traces.py",
        "Collecting routing traces from Switch Transformer"
    )
    
    if success:
        # Verify traces were created
        if trace_file.exists():
            size_mb = trace_file.stat().st_size / (1024*1024)
            logger.info(f"✅ Traces collected: {size_mb:.1f} MB")
            return True
        else:
            logger.error("❌ Trace file not found after collection")
            return False
    
    return False

def step2_train_models(args):
    """Step 2: Train speculation models on collected traces"""
    logger.info("=" * 60)
    logger.info("STEP 2: TRAINING SPECULATION MODELS")
    logger.info("=" * 60)
    
    # Check if trained models already exist
    model_files = [
        "trained_models/contextual_real_data.pt",
        "trained_models/transformer_real_data.pt", 
        "trained_models/hierarchical_real_data.pt"
    ]
    
    if all(Path(f).exists() for f in model_files) and not args.force_retrain:
        logger.info("✅ Trained models already exist")
        logger.info("Use --force-retrain to retrain models")
        return True
    
    # Run training
    success = run_command(
        "python train_with_real_traces.py",
        "Training learnable gating models on real traces"
    )
    
    if success:
        # Verify models were created
        created_models = [f for f in model_files if Path(f).exists()]
        logger.info(f"✅ Models trained: {len(created_models)} / {len(model_files)}")
        return len(created_models) > 0
    
    return False

def step3_test_speculation(args):
    """Step 3: Test speculation models with current infrastructure"""
    logger.info("=" * 60)
    logger.info("STEP 3: TESTING SPECULATION MODELS")
    logger.info("=" * 60)
    
    # Test with custom model first
    logger.info("Testing with custom small model...")
    success_custom = run_command(
        "python main.py --mode demo --speculation-mode multi_layer",
        "Testing custom model with speculation"
    )
    
    # Test with pre-trained model if available
    logger.info("Testing with pre-trained Switch Transformer...")
    success_pretrained = run_command(
        "python main_pretrained.py --mode demo --pretrained-model google/switch-base-8",
        "Testing pre-trained model with speculation"
    )
    
    # Run comparison of speculation modes
    logger.info("Comparing speculation modes...")
    success_compare = run_command(
        "python main.py --mode compare",
        "Comparing different speculation strategies"
    )
    
    return success_custom or success_pretrained

def step4_evaluate_performance(args):
    """Step 4: Comprehensive evaluation and benchmarking"""
    logger.info("=" * 60)
    logger.info("STEP 4: COMPREHENSIVE EVALUATION")
    logger.info("=" * 60)
    
    # Run speculation benchmark
    success_benchmark = run_command(
        "python run_speculation_experiments.py",
        "Running comprehensive speculation benchmark"
    )
    
    # Generate final report
    success_status = run_command(
        "python check_current_status.py",
        "Generating final status report"
    )
    
    return success_benchmark or success_status

def step5_integration_test(args):
    """Step 5: Test learnable models with speculation engine"""
    logger.info("=" * 60)
    logger.info("STEP 5: INTEGRATION TEST - LEARNABLE SPECULATION")
    logger.info("=" * 60)
    
    # Test if we can load and use trained models
    integration_script = """
import torch
import sys
from pathlib import Path

# Load trained model if available
model_path = Path("trained_models/contextual_real_data.pt")
if model_path.exists():
    print(f"✅ Loading trained model: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"✅ Model loaded successfully")
    print(f"Training stats: {checkpoint.get('training_stats', {})}")
    
    # Test model integration with speculation engine
    from training.learnable_gating_models import create_gating_model, GatingModelConfig
    from gating.speculation_engine import LearnableSpeculationEngine
    
    config = GatingModelConfig(
        hidden_size=checkpoint['gating_config']['hidden_size'],
        num_experts=checkpoint['gating_config']['num_experts'],
        num_layers=checkpoint['gating_config']['num_layers']
    )
    
    model = create_gating_model("contextual", config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model architecture loaded and ready")
    
    # Create learnable speculation engine
    speculation_engine = LearnableSpeculationEngine(
        num_experts=config.num_experts,
        num_layers=config.num_layers,
        gating_model=model
    )
    print("✅ Learnable speculation engine created")
    
    # Test prediction
    import torch
    hidden_states = torch.randn(1, 32, config.hidden_size)
    prev_gates = [torch.randn(1, 32, config.num_experts) for _ in range(3)]
    
    predicted_experts, confidence = speculation_engine.predict_next_experts(
        current_layer=2,
        hidden_states=hidden_states,
        current_routing=prev_gates[-1] if prev_gates else None
    )
    
    print(f"✅ Prediction successful: {predicted_experts.shape}, confidence: {confidence.mean():.3f}")
    print("🎉 INTEGRATION TEST PASSED!")
    
else:
    print("❌ No trained models found. Run training step first.")
    sys.exit(1)
"""
    
    # Write and run integration test
    test_file = Path("test_integration.py")
    with open(test_file, 'w') as f:
        f.write(integration_script)
    
    try:
        success = run_command(
            "python test_integration.py",
            "Testing learnable model integration"
        )
        
        # Clean up test file
        test_file.unlink()
        return success
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        if test_file.exists():
            test_file.unlink()
        return False

def main():
    parser = argparse.ArgumentParser(description="Enhanced Pre-gated MoE Speculation Pipeline")
    parser.add_argument('--steps', nargs='+', choices=['1', '2', '3', '4', '5', 'all'], 
                       default=['all'], help='Pipeline steps to run')
    parser.add_argument('--force-recollect', action='store_true', 
                       help='Force re-collection of traces even if they exist')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force re-training of models even if they exist')
    parser.add_argument('--skip-prerequisites', action='store_true',
                       help='Skip prerequisite checks')
    
    args = parser.parse_args()
    
    print("🚀 ENHANCED PRE-GATED MOE SPECULATION PIPELINE")
    print("=" * 70)
    print("Complete workflow: Trace Collection → Training → Testing → Evaluation")
    print("=" * 70)
    
    start_time = time.time()
    
    # Check prerequisites
    if not args.skip_prerequisites:
        if not check_prerequisites():
            logger.error("❌ Prerequisites not met. Exiting.")
            sys.exit(1)
    
    # Determine which steps to run
    if 'all' in args.steps:
        steps_to_run = ['1', '2', '3', '4', '5']
    else:
        steps_to_run = args.steps
    
    results = {}
    
    # Execute pipeline steps
    if '1' in steps_to_run:
        results['step1'] = step1_collect_traces(args)
    
    if '2' in steps_to_run:
        results['step2'] = step2_train_models(args)
    
    if '3' in steps_to_run:
        results['step3'] = step3_test_speculation(args)
    
    if '4' in steps_to_run:
        results['step4'] = step4_evaluate_performance(args)
    
    if '5' in steps_to_run:
        results['step5'] = step5_integration_test(args)
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🎯 PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    
    for step, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        step_name = {
            'step1': 'Trace Collection',
            'step2': 'Model Training', 
            'step3': 'Speculation Testing',
            'step4': 'Performance Evaluation',
            'step5': 'Integration Test'
        }.get(step, step)
        print(f"{step_name:<25} {status}")
    
    successful_steps = sum(results.values())
    total_steps = len(results)
    
    print("-" * 70)
    print(f"Overall Success Rate: {successful_steps}/{total_steps} ({successful_steps/total_steps*100:.1f}%)")
    print(f"Total Execution Time: {total_time:.1f} seconds")
    
    if successful_steps == total_steps:
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("✅ Your Enhanced Pre-gated MoE system is ready for use!")
        print("\nNext steps:")
        print("  • Run 'python main.py --mode demo' for basic testing")
        print("  • Run 'python main_pretrained.py --mode demo' for pre-trained models")
        print("  • Check 'trained_models/' for your learnable gating models")
    else:
        print(f"\n⚠️  PIPELINE PARTIALLY COMPLETED ({successful_steps}/{total_steps} steps)")
        print("Check the logs above for specific failures and retry those steps.")
    
    return successful_steps == total_steps

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)