#!/usr/bin/env python3
"""
Comprehensive Comparison of All Speculation Approaches
Tests all implemented strategies and reports best performance
"""

import torch
import logging
import time
from pathlib import Path
import subprocess
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeculationBenchmark:
    """Comprehensive benchmark of all speculation approaches"""
    
    def __init__(self):
        self.results = {}
        self.training_times = {}
        
    def run_all_approaches(self):
        """Run all implemented approaches"""
        
        logger.info("🏁 Comprehensive Speculation Approach Comparison")
        logger.info("=" * 70)
        
        approaches = [
            {
                'name': 'Baseline Simple',
                'script': 'train_simple_fixed.py',
                'description': 'Basic single-model approach'
            },
            {
                'name': 'Improved Strategies',
                'script': 'train_improved_accuracy.py',
                'description': 'Weighted + sequence-aware + ensemble strategies'
            },
            {
                'name': 'Sophisticated Multi-Layer',
                'script': 'train_sophisticated_speculation.py',
                'description': 'Layer-specific models + multi-step prediction'
            }
        ]
        
        for approach in approaches:
            logger.info(f"\n{'='*50}")
            logger.info(f"🧪 TESTING: {approach['name']}")
            logger.info(f"📝 {approach['description']}")
            logger.info(f"{'='*50}")
            
            try:
                start_time = time.time()
                
                # Run the training script
                result = subprocess.run([
                    'python', approach['script']
                ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
                
                training_time = time.time() - start_time
                self.training_times[approach['name']] = training_time
                
                if result.returncode == 0:
                    # Parse results from output
                    accuracy = self._parse_accuracy(result.stdout)
                    self.results[approach['name']] = {
                        'accuracy': accuracy,
                        'status': 'success',
                        'training_time': training_time,
                        'output': result.stdout
                    }
                    logger.info(f"✅ {approach['name']} completed: {accuracy:.3f} accuracy")
                else:
                    self.results[approach['name']] = {
                        'accuracy': 0,
                        'status': 'failed',
                        'training_time': training_time,
                        'error': result.stderr
                    }
                    logger.error(f"❌ {approach['name']} failed")
                    
            except subprocess.TimeoutExpired:
                self.results[approach['name']] = {
                    'accuracy': 0,
                    'status': 'timeout',
                    'training_time': 1800,
                    'error': 'Training timeout'
                }
                logger.error(f"⏰ {approach['name']} timed out")
                
            except Exception as e:
                self.results[approach['name']] = {
                    'accuracy': 0,
                    'status': 'error',
                    'training_time': 0,
                    'error': str(e)
                }
                logger.error(f"💥 {approach['name']} error: {e}")
        
        return self._generate_comparison_report()
    
    def _parse_accuracy(self, output):
        """Parse accuracy from training output"""
        try:
            # Look for patterns like "Best validation accuracy: 0.148 (14.8%)"
            lines = output.split('\n')
            for line in lines:
                if 'Best validation accuracy:' in line:
                    # Extract number between 'accuracy:' and '('
                    parts = line.split('accuracy:')[1].split('(')[0].strip()
                    return float(parts)
            return 0
        except:
            return 0
    
    def _generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        
        report = []
        report.append("🏆 COMPREHENSIVE SPECULATION COMPARISON RESULTS")
        report.append("=" * 80)
        
        # Sort by accuracy
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1].get('accuracy', 0), 
                               reverse=True)
        
        # Performance ranking
        report.append("\n📊 PERFORMANCE RANKING:")
        report.append("-" * 40)
        
        for i, (name, result) in enumerate(sorted_results):
            if result.get('accuracy', 0) > 0:
                status_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                accuracy = result['accuracy']
                improvement = accuracy / (1/128)  # vs random baseline
                time_min = result['training_time'] / 60
                
                report.append(f"{status_icon} {i+1}. {name:25} | "
                             f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%) | "
                             f"Improvement: {improvement:.1f}x | "
                             f"Time: {time_min:.1f}min")
            else:
                report.append(f"❌ {i+1}. {name:25} | FAILED: {result.get('status', 'unknown')}")
        
        # Technical comparison
        report.append(f"\n🔬 TECHNICAL COMPARISON:")
        report.append("-" * 30)
        
        if sorted_results:
            best_name, best_result = sorted_results[0]
            baseline_accuracy = 1/128
            
            report.append(f"Best approach: {best_name}")
            report.append(f"Best accuracy: {best_result.get('accuracy', 0):.3f} ({best_result.get('accuracy', 0)*100:.1f}%)")
            report.append(f"Random baseline: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
            report.append(f"Total improvement: {(best_result.get('accuracy', 0)/baseline_accuracy):.1f}x better than random")
        
        # Efficiency analysis
        report.append(f"\n⚡ EFFICIENCY ANALYSIS:")
        report.append("-" * 25)
        
        for name, result in sorted_results:
            if result.get('accuracy', 0) > 0:
                accuracy = result['accuracy']
                time_sec = result['training_time']
                efficiency = (accuracy * 1000) / time_sec  # accuracy per second * 1000
                
                report.append(f"{name:25} | Efficiency: {efficiency:.2f} (acc*1000/sec)")
        
        # Architecture insights
        report.append(f"\n🏗️ ARCHITECTURE INSIGHTS:")
        report.append("-" * 28)
        
        insights = {
            'Baseline Simple': 'Single model, mean pooling, basic architecture',
            'Improved Strategies': 'Multi-scale features, weighted targets, label smoothing',
            'Sophisticated Multi-Layer': 'Layer-specific models, multi-step prediction, ensemble'
        }
        
        for name, insight in insights.items():
            if name in self.results:
                status = "✅" if self.results[name].get('accuracy', 0) > 0 else "❌"
                report.append(f"{status} {name:25} | {insight}")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        report.append("-" * 20)
        
        if sorted_results and sorted_results[0][1].get('accuracy', 0) > 0:
            best_approach = sorted_results[0][0]
            best_acc = sorted_results[0][1]['accuracy']
            
            if best_acc > 0.3:  # 30%
                report.append("🎯 EXCELLENT: Ready for production deployment!")
                report.append(f"   Use {best_approach} for speculative loading")
                
            elif best_acc > 0.2:  # 20%
                report.append("🎯 GOOD: Strong speculation capability")
                report.append(f"   Deploy {best_approach} with confidence monitoring")
                
            elif best_acc > 0.1:  # 10%
                report.append("🎯 MODERATE: Useful speculation but needs improvement")
                report.append(f"   Consider {best_approach} with additional optimizations")
                
            else:
                report.append("🎯 LOW: Needs significant improvement")
                report.append("   Consider more training data or architecture changes")
            
            # Next steps
            report.append(f"\n🚀 NEXT STEPS:")
            report.append("1. Deploy best model for speculative expert loading")
            report.append("2. Measure real-world speedup on MoE inference")
            report.append("3. Implement multi-layer lookahead (N+1, N+2, N+3)")
            report.append("4. Scale to larger datasets (10,000+ traces)")
            report.append("5. Test on edge devices (Jetson)")
        
        # Save detailed results
        self._save_detailed_results()
        
        report_text = "\n".join(report)
        
        # Save report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"comparison_reports/speculation_comparison_{timestamp}.txt"
        Path("comparison_reports").mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"📄 Detailed report saved to: {report_file}")
        
        return report_text
    
    def _save_detailed_results(self):
        """Save detailed results as JSON"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"comparison_reports/detailed_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"📊 Detailed results saved to: {results_file}")

def main():
    """Run comprehensive comparison"""
    
    # Ensure directories exist
    Path("trained_models").mkdir(exist_ok=True)
    Path("comparison_reports").mkdir(exist_ok=True)
    
    # Run benchmark
    benchmark = SpeculationBenchmark()
    report = benchmark.run_all_approaches()
    
    # Print final report
    print("\n" + report)
    
    logger.info("🎉 Comprehensive comparison completed!")
    
    return True

if __name__ == "__main__":
    main()