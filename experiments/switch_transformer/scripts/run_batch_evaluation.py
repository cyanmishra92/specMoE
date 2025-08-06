#!/usr/bin/env python3
"""
Run Batch Evaluation of All Switch Prefetching Experiments

Executes the full 5×5 experiment matrix:
- 5 strategies: A, B, C, D, E  
- 5 batch sizes: 1, 2, 4, 8, 16
- Multiple runs per experiment for statistical significance
"""

import subprocess
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BatchEvaluator:
    def __init__(self, runs_per_experiment: int = 10, max_parallel: int = 2):
        self.runs_per_experiment = runs_per_experiment
        self.max_parallel = max_parallel
        self.strategies = ["A", "B", "C", "D", "E"]
        self.batch_sizes = [1, 2, 4, 8, 16]
        self.total_experiments = len(self.strategies) * len(self.batch_sizes)
        
        logging.info(f"🚀 Batch Evaluator initialized:")
        logging.info(f"   Strategies: {self.strategies}")
        logging.info(f"   Batch sizes: {self.batch_sizes}")
        logging.info(f"   Total experiments: {self.total_experiments}")
        logging.info(f"   Runs per experiment: {runs_per_experiment}")
        logging.info(f"   Max parallel: {max_parallel}")
    
    def run_single_experiment(self, strategy: str, batch_size: int, output_dir: Path) -> dict:
        """Run a single experiment and return results"""
        experiment_name = f"Strategy_{strategy}_Batch_{batch_size}"
        start_time = time.time()
        
        try:
            # Build command
            cmd = [
                "python", "run_single_experiment.py",
                "--strategy", strategy,
                "--batch-size", str(batch_size),
                "--runs", str(self.runs_per_experiment),
                "--output-dir", str(output_dir)
            ]
            
            logging.info(f"▶️  Starting {experiment_name}")
            
            # Run experiment
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout per experiment
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                logging.info(f"✅ {experiment_name} completed in {elapsed_time:.1f}s")
                return {
                    "strategy": strategy,
                    "batch_size": batch_size,
                    "status": "success",
                    "elapsed_time": elapsed_time,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                logging.error(f"❌ {experiment_name} failed (code {result.returncode})")
                logging.error(f"   Error: {result.stderr}")
                return {
                    "strategy": strategy,
                    "batch_size": batch_size,
                    "status": "failed",
                    "elapsed_time": elapsed_time,
                    "error": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            logging.error(f"⏰ {experiment_name} timed out after 30 minutes")
            return {
                "strategy": strategy,
                "batch_size": batch_size,
                "status": "timeout",
                "elapsed_time": 1800
            }
        except Exception as e:
            logging.error(f"💥 {experiment_name} crashed: {str(e)}")
            return {
                "strategy": strategy,
                "batch_size": batch_size,
                "status": "error",
                "elapsed_time": time.time() - start_time,
                "error": str(e)
            }
    
    def run_all_experiments(self, output_dir: Path, parallel: bool = True) -> dict:
        """Run all experiments in the matrix"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment list
        experiments = [
            (strategy, batch_size) 
            for strategy in self.strategies 
            for batch_size in self.batch_sizes
        ]
        
        logging.info(f"🎯 Starting evaluation of {len(experiments)} experiments")
        start_time = time.time()
        
        results = []
        
        if parallel and self.max_parallel > 1:
            # Run experiments in parallel
            with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                # Submit all experiments
                future_to_exp = {
                    executor.submit(self.run_single_experiment, strategy, batch_size, output_dir): (strategy, batch_size)
                    for strategy, batch_size in experiments
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_exp):
                    strategy, batch_size = future_to_exp[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        completed = len(results)
                        progress = completed / len(experiments) * 100
                        logging.info(f"📊 Progress: {completed}/{len(experiments)} ({progress:.1f}%)")
                        
                    except Exception as e:
                        logging.error(f"💥 Experiment {strategy}_{batch_size} generated exception: {e}")
                        results.append({
                            "strategy": strategy,
                            "batch_size": batch_size,
                            "status": "exception",
                            "error": str(e)
                        })
        else:
            # Run experiments sequentially
            for i, (strategy, batch_size) in enumerate(experiments):
                result = self.run_single_experiment(strategy, batch_size, output_dir)
                results.append(result)
                
                progress = (i + 1) / len(experiments) * 100
                logging.info(f"📊 Progress: {i + 1}/{len(experiments)} ({progress:.1f}%)")
        
        total_time = time.time() - start_time
        
        # Compile summary
        summary = {
            "total_experiments": len(experiments),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "failed"]),
            "timeout": len([r for r in results if r["status"] == "timeout"]),
            "errors": len([r for r in results if r["status"] == "error"]),
            "total_time_hours": total_time / 3600,
            "average_time_per_experiment": total_time / len(experiments),
            "results": results
        }
        
        return summary
    
    def save_summary(self, summary: dict, output_file: Path):
        """Save evaluation summary"""
        import json
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logging.info(f"📁 Summary saved to: {output_file}")
    
    def print_summary(self, summary: dict):
        """Print evaluation summary"""
        print(f"\n🎯 BATCH EVALUATION SUMMARY")
        print(f"=" * 50)
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Timeout: {summary['timeout']}")
        print(f"Errors: {summary['errors']}")
        print(f"Total time: {summary['total_time_hours']:.1f} hours")
        print(f"Avg time per experiment: {summary['average_time_per_experiment']:.1f} seconds")
        
        # Show failed experiments
        failed_experiments = [r for r in summary['results'] if r['status'] != 'success']
        if failed_experiments:
            print(f"\n❌ Failed Experiments:")
            for exp in failed_experiments:
                print(f"   {exp['strategy']}_{exp['batch_size']}: {exp['status']}")
        
        print(f"\n✅ Evaluation completed!")

def create_evaluation_script(output_dir: Path):
    """Create convenient evaluation script"""
    script_content = f'''#!/bin/bash
# Switch Transformer Evaluation Runner
# Generated automatically

echo "🚀 Starting Switch Transformer Evaluation"
echo "Total experiments: 25 (5 strategies × 5 batch sizes)"
echo "Expected time: 2-4 hours"
echo ""

# Create results directory
mkdir -p {output_dir}

# Run evaluation
python run_batch_evaluation.py \\
    --runs 10 \\
    --parallel \\
    --max-parallel 2 \\
    --output-dir {output_dir}

echo ""
echo "✅ Evaluation completed!"
echo "Results available in: {output_dir}"
echo ""
echo "Next steps:"
echo "1. python analyze_results.py --input {output_dir}"
echo "2. python generate_csv_report.py --input {output_dir}"
'''
    
    script_file = output_dir.parent / "run_evaluation.sh"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_file.chmod(0o755)
    print(f"📜 Evaluation script created: {script_file}")

def main():
    parser = argparse.ArgumentParser(description="Batch Switch Prefetching Evaluation")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of runs per experiment")
    parser.add_argument("--parallel", action="store_true",
                       help="Run experiments in parallel")
    parser.add_argument("--max-parallel", type=int, default=2,
                       help="Maximum parallel experiments")
    parser.add_argument("--output-dir", type=str, default="../results",
                       help="Output directory")
    parser.add_argument("--create-script", action="store_true",
                       help="Create evaluation script and exit")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if args.create_script:
        create_evaluation_script(output_dir)
        return
    
    # Initialize evaluator
    evaluator = BatchEvaluator(
        runs_per_experiment=args.runs,
        max_parallel=args.max_parallel if args.parallel else 1
    )
    
    # Run all experiments
    summary = evaluator.run_all_experiments(output_dir, args.parallel)
    
    # Save and print summary
    summary_file = output_dir / "evaluation_summary.json"
    evaluator.save_summary(summary, summary_file)
    evaluator.print_summary(summary)
    
    # Create analysis script
    if summary['successful'] > 0:
        print(f"\n📊 Next steps:")
        print(f"1. python analyze_results.py --input {output_dir}")
        print(f"2. python generate_csv_report.py --input {output_dir}")

if __name__ == "__main__":
    main()