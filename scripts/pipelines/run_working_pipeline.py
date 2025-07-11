#!/usr/bin/env python3
"""
CORRECTED Working Pipeline - Uses Original Working Scripts
Uses the scripts that were actually working before my refactoring
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
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed")
        logger.error(f"Error: {e.stderr}")
        if e.stdout:
            print(e.stdout)
        return False

def collect_traces_with_128_experts(args):
    """Use the working 128-expert trace collector"""
    logger.info("=" * 60)
    logger.info("COLLECTING TRACES WITH 128 EXPERT MODEL")
    logger.info("=" * 60)
    
    # Check if we should use the robust collector (128 experts)
    trace_file = Path("routing_data/robust_traces.pkl")
    
    if trace_file.exists() and not args.force_recollect:
        size_mb = trace_file.stat().st_size / (1024*1024)
        logger.info(f"✅ 128-expert traces already exist: {trace_file} ({size_mb:.1f} MB)")
        logger.info("Use --force-recollect to regenerate traces")
        return True
    
    # Run the working 128-expert collector
    logger.info("🔥 Running 128-expert Switch Transformer trace collection...")
    logger.info("This will take time as it processes the large model...")
    
    success = run_command(
        "python scripts/collection/collect_robust_traces.py",
        "Collecting traces with google/switch-base-128"
    )
    
    if success and trace_file.exists():
        size_mb = trace_file.stat().st_size / (1024*1024)
        logger.info(f"✅ 128-expert traces collected: {size_mb:.1f} MB")
    
    return success

def collect_working_final_traces(args):
    """Use the final working collector"""
    logger.info("=" * 60)
    logger.info("COLLECTING WITH FINAL WORKING COLLECTOR")
    logger.info("=" * 60)
    
    trace_file = Path("routing_data/working_final_traces.pkl")
    
    if trace_file.exists() and not args.force_recollect:
        size_mb = trace_file.stat().st_size / (1024*1024)
        logger.info(f"✅ Working final traces already exist: {trace_file} ({size_mb:.1f} MB)")
        return True
    
    success = run_command(
        "python scripts/collection/collect_working_final.py",
        "Collecting traces with final working collector"
    )
    
    return success

def run_proper_training_testing(args):
    """Use the proper train/test script"""
    logger.info("=" * 60)
    logger.info("PROPER TRAINING AND TESTING")
    logger.info("=" * 60)
    
    success = run_command(
        "python scripts/training/proper_train_test.py",
        "Running proper train/test with robust data splits"
    )
    
    return success

def test_individual_approaches(args):
    """Test individual speculation approaches"""
    logger.info("=" * 60)
    logger.info("TESTING INDIVIDUAL APPROACHES")
    logger.info("=" * 60)
    
    success = run_command(
        "python scripts/evaluation/test_individual_approaches.py",
        "Testing individual speculation approaches"
    )
    
    return success

def compare_all_approaches(args):
    """Compare all approaches comprehensively"""
    logger.info("=" * 60)
    logger.info("COMPARING ALL APPROACHES")
    logger.info("=" * 60)
    
    success = run_command(
        "python scripts/evaluation/compare_all_approaches.py",
        "Comprehensive comparison of all speculation approaches"
    )
    
    return success

def main():
    parser = argparse.ArgumentParser(description="CORRECTED Working Speculation Pipeline")
    parser.add_argument('--steps', nargs='+', choices=['1', '2', '3', '4', '5', 'all'], 
                       default=['all'], help='Pipeline steps to run')
    parser.add_argument('--force-recollect', action='store_true', 
                       help='Force re-collection of traces even if they exist')
    parser.add_argument('--use-128-experts', action='store_true',
                       help='Use 128-expert model for trace collection (takes longer)')
    
    args = parser.parse_args()
    
    print("🔧 CORRECTED SPECULATION PIPELINE")
    print("=" * 70)
    print("Using the ORIGINAL WORKING scripts that were moved to archive")
    print("=" * 70)
    
    start_time = time.time()
    
    # Determine which steps to run
    if 'all' in args.steps:
        steps_to_run = ['1', '2', '3', '4', '5']
    else:
        steps_to_run = args.steps
    
    results = {}
    
    # Step 1: Collect traces (choose collector)
    if '1' in steps_to_run:
        if args.use_128_experts:
            results['step1'] = collect_traces_with_128_experts(args)
        else:
            results['step1'] = collect_working_final_traces(args)
    
    # Step 2: Proper training and testing
    if '2' in steps_to_run:
        results['step2'] = run_proper_training_testing(args)
    
    # Step 3: Test individual approaches
    if '3' in steps_to_run:
        results['step3'] = test_individual_approaches(args)
    
    # Step 4: Compare all approaches
    if '4' in steps_to_run:
        results['step4'] = compare_all_approaches(args)
    
    # Step 5: Status check
    if '5' in steps_to_run:
        results['step5'] = run_command(
            "python scripts/check_current_status.py",
            "Final status check"
        )
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🎯 CORRECTED PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    
    step_names = {
        'step1': 'Trace Collection (Working Scripts)',
        'step2': 'Proper Training & Testing',
        'step3': 'Individual Approach Testing',
        'step4': 'Comprehensive Comparison',
        'step5': 'Status Check'
    }
    
    for step, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        step_name = step_names.get(step, step)
        print(f"{step_name:<35} {status}")
    
    successful_steps = sum(results.values())
    total_steps = len(results)
    
    print("-" * 70)
    print(f"Overall Success Rate: {successful_steps}/{total_steps} ({successful_steps/total_steps*100:.1f}%)")
    print(f"Total Execution Time: {total_time:.1f} seconds")
    
    if successful_steps == total_steps:
        print("\n🎉 CORRECTED PIPELINE COMPLETED SUCCESSFULLY!")
        print("✅ Using the original working scripts restored functionality!")
        print("\nWorking scripts used:")
        print("  • collect_robust_traces.py (128 experts)")
        print("  • collect_working_final.py (confirmed working)")
        print("  • proper_train_test.py (proper data splits)")
        print("  • test_individual_approaches.py")
        print("  • compare_all_approaches.py")
    else:
        print(f"\n⚠️  PIPELINE PARTIALLY COMPLETED ({successful_steps}/{total_steps} steps)")
        print("The working scripts should resolve the previous issues.")
    
    return successful_steps == total_steps

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)