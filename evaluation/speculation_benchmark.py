"""
Comprehensive Speculation Accuracy Benchmarking Framework
Measures multiple aspects of speculation quality and effectiveness
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import time
import json
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gating.speculation_engine import SpeculationEngine, SpeculationMode, InputType


@dataclass
class SpeculationMetrics:
    """Comprehensive metrics for speculation accuracy"""
    
    # Basic accuracy metrics
    top_k_accuracy: Dict[int, float]  # Top-1, top-2, top-3 accuracy
    probability_correlation: float    # Correlation between predicted and actual probs
    kl_divergence: float             # KL divergence between distributions
    
    # Confidence-based metrics
    confidence_accuracy_correlation: float  # High confidence → high accuracy?
    calibration_error: float                # How well calibrated are confidence scores?
    
    # Temporal stability metrics
    prediction_stability: float      # How stable are predictions over time?
    adaptation_speed: float          # How fast does system adapt to changes?
    
    # Efficiency metrics
    speculation_overhead: float      # Time overhead of speculation
    memory_overhead: float          # Memory overhead
    
    # Hardware-specific metrics
    gpu_utilization: float          # GPU utilization during speculation
    memory_bandwidth_efficiency: float  # Memory bandwidth usage


class SpeculationBenchmark:
    """
    Comprehensive benchmarking suite for speculation accuracy and efficiency
    """
    
    def __init__(self, model, speculation_engine: SpeculationEngine, device: str = "cuda"):
        self.model = model
        self.speculation_engine = speculation_engine
        self.device = device
        
        # Data collection
        self.prediction_history = defaultdict(list)
        self.actual_history = defaultdict(list)
        self.confidence_history = defaultdict(list)
        self.timing_data = defaultdict(list)
        
        # Metrics storage
        self.metrics = {}
        self.layer_metrics = defaultdict(dict)
        
    def run_comprehensive_benchmark(
        self, 
        test_inputs: List[torch.Tensor], 
        input_types: List[str] = None,
        save_results: bool = True
    ) -> Dict:
        """Run complete benchmark suite"""
        
        print("🔍 Running Comprehensive Speculation Benchmark")
        print("=" * 50)
        
        # 1. Accuracy Benchmarks
        print("1. Measuring Speculation Accuracy...")
        accuracy_metrics = self._benchmark_accuracy(test_inputs, input_types)
        
        # 2. Confidence Calibration
        print("2. Evaluating Confidence Calibration...")
        calibration_metrics = self._benchmark_confidence_calibration(test_inputs)
        
        # 3. Temporal Stability
        print("3. Analyzing Temporal Stability...")
        stability_metrics = self._benchmark_temporal_stability(test_inputs)
        
        # 4. Mode Comparison
        print("4. Comparing Speculation Modes...")
        mode_comparison = self._benchmark_speculation_modes(test_inputs)
        
        # 5. Hardware Efficiency
        print("5. Measuring Hardware Efficiency...")
        efficiency_metrics = self._benchmark_hardware_efficiency(test_inputs)
        
        # Combine all results
        results = {
            'accuracy': accuracy_metrics,
            'calibration': calibration_metrics,
            'stability': stability_metrics,
            'mode_comparison': mode_comparison,
            'efficiency': efficiency_metrics,
            'timestamp': time.time()
        }
        
        if save_results:
            self._save_benchmark_results(results)
            
        return results
    
    def _benchmark_accuracy(self, test_inputs: List[torch.Tensor], input_types: List[str] = None) -> Dict:
        """Comprehensive accuracy benchmarking"""
        
        metrics = {
            'top_k_accuracies': {1: [], 2: [], 3: []},
            'probability_correlations': [],
            'kl_divergences': [],
            'per_layer_accuracy': defaultdict(list),
            'per_input_type_accuracy': defaultdict(list)
        }
        
        self.speculation_engine.reset_statistics()
        
        for i, input_tensor in enumerate(test_inputs):
            input_type = input_types[i] if input_types else "unknown"
            
            # Run model with speculation
            with torch.no_grad():
                # Simulate layer-by-layer processing
                for layer_id in range(self.speculation_engine.num_layers - 1):
                    # Get prediction
                    pred_probs, confidence = self.speculation_engine.predict_next_experts(
                        layer_id, input_tensor
                    )
                    
                    # Simulate actual routing (would come from model)
                    actual_probs = F.softmax(torch.randn(self.speculation_engine.num_experts), dim=0)
                    
                    # Calculate multiple accuracy metrics
                    top_k_accs = self._calculate_top_k_accuracy(pred_probs, actual_probs)
                    prob_corr = self._calculate_probability_correlation(pred_probs, actual_probs)
                    kl_div = self._calculate_kl_divergence(pred_probs, actual_probs)
                    
                    # Store metrics
                    for k in [1, 2, 3]:
                        metrics['top_k_accuracies'][k].append(top_k_accs[k])
                    metrics['probability_correlations'].append(prob_corr)
                    metrics['kl_divergences'].append(kl_div)
                    metrics['per_layer_accuracy'][layer_id].append(top_k_accs[1])
                    metrics['per_input_type_accuracy'][input_type].append(top_k_accs[1])
                    
                    # Update speculation engine
                    self.speculation_engine.update_accuracy(layer_id, pred_probs, actual_probs.unsqueeze(0))
        
        # Aggregate results
        return {
            'mean_top_k_accuracy': {k: np.mean(v) for k, v in metrics['top_k_accuracies'].items()},
            'std_top_k_accuracy': {k: np.std(v) for k, v in metrics['top_k_accuracies'].items()},
            'mean_probability_correlation': np.mean(metrics['probability_correlations']),
            'mean_kl_divergence': np.mean(metrics['kl_divergences']),
            'per_layer_accuracy': {k: np.mean(v) for k, v in metrics['per_layer_accuracy'].items()},
            'per_input_type_accuracy': {k: np.mean(v) for k, v in metrics['per_input_type_accuracy'].items()},
            'detailed_metrics': metrics
        }
    
    def _benchmark_confidence_calibration(self, test_inputs: List[torch.Tensor]) -> Dict:
        """Measure how well confidence scores match actual accuracy"""
        
        confidence_bins = np.linspace(0, 1, 11)  # 10 bins
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        predictions = []
        confidences = []
        accuracies = []
        
        for input_tensor in test_inputs:
            for layer_id in range(self.speculation_engine.num_layers - 1):
                pred_probs, confidence = self.speculation_engine.predict_next_experts(
                    layer_id, input_tensor
                )
                actual_probs = F.softmax(torch.randn(self.speculation_engine.num_experts), dim=0)
                
                # Calculate accuracy for this prediction
                top_1_acc = self._calculate_top_k_accuracy(pred_probs, actual_probs)[1]
                
                predictions.append(pred_probs)
                confidences.append(confidence)
                accuracies.append(top_1_acc)
        
        # Bin predictions by confidence
        for i in range(len(confidence_bins) - 1):
            bin_mask = (np.array(confidences) >= confidence_bins[i]) & (np.array(confidences) < confidence_bins[i + 1])
            
            if np.sum(bin_mask) > 0:
                bin_acc = np.mean(np.array(accuracies)[bin_mask])
                bin_conf = np.mean(np.array(confidences)[bin_mask])
                bin_count = np.sum(bin_mask)
                
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(bin_count)
        
        # Calculate calibration error (ECE - Expected Calibration Error)
        ece = 0.0
        total_samples = len(confidences)
        for i in range(len(bin_accuracies)):
            ece += (bin_counts[i] / total_samples) * abs(bin_accuracies[i] - bin_confidences[i])
        
        return {
            'expected_calibration_error': ece,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'confidence_accuracy_correlation': np.corrcoef(confidences, accuracies)[0, 1] if len(confidences) > 1 else 0.0
        }
    
    def _benchmark_temporal_stability(self, test_inputs: List[torch.Tensor]) -> Dict:
        """Measure prediction stability over time"""
        
        stability_metrics = {
            'prediction_variance': [],
            'confidence_variance': [],
            'adaptation_times': [],
            'stability_per_layer': defaultdict(list)
        }
        
        for input_tensor in test_inputs:
            layer_predictions = defaultdict(list)
            layer_confidences = defaultdict(list)
            
            # Multiple passes to measure stability
            for pass_id in range(10):
                for layer_id in range(self.speculation_engine.num_layers - 1):
                    pred_probs, confidence = self.speculation_engine.predict_next_experts(
                        layer_id, input_tensor
                    )
                    
                    layer_predictions[layer_id].append(pred_probs.numpy())
                    layer_confidences[layer_id].append(confidence)
            
            # Calculate variance metrics
            for layer_id in layer_predictions:
                pred_array = np.stack(layer_predictions[layer_id])
                pred_variance = np.mean(np.var(pred_array, axis=0))
                conf_variance = np.var(layer_confidences[layer_id])
                
                stability_metrics['prediction_variance'].append(pred_variance)
                stability_metrics['confidence_variance'].append(conf_variance)
                stability_metrics['stability_per_layer'][layer_id].append(pred_variance)
        
        return {
            'mean_prediction_variance': np.mean(stability_metrics['prediction_variance']),
            'mean_confidence_variance': np.mean(stability_metrics['confidence_variance']),
            'stability_per_layer': {k: np.mean(v) for k, v in stability_metrics['stability_per_layer'].items()}
        }
    
    def _benchmark_speculation_modes(self, test_inputs: List[torch.Tensor]) -> Dict:
        """Compare different speculation modes"""
        
        modes = [
            SpeculationMode.LAYER_MINUS_1,
            SpeculationMode.MULTI_LAYER,
            SpeculationMode.PATTERN_LEARNING,
            SpeculationMode.ADAPTIVE
        ]
        
        mode_results = {}
        
        for mode in modes:
            print(f"  Testing {mode.value}...")
            
            # Create engine with this mode
            engine = SpeculationEngine(
                num_experts=self.speculation_engine.num_experts,
                num_layers=self.speculation_engine.num_layers,
                speculation_mode=mode
            )
            
            # Run mini benchmark
            accuracies = []
            confidences = []
            timings = []
            
            for input_tensor in test_inputs[:5]:  # Subset for speed
                for layer_id in range(engine.num_layers - 1):
                    start_time = time.time()
                    pred_probs, confidence = engine.predict_next_experts(layer_id, input_tensor)
                    end_time = time.time()
                    
                    # Simulate actual routing
                    actual_probs = F.softmax(torch.randn(engine.num_experts), dim=0)
                    accuracy = self._calculate_top_k_accuracy(pred_probs, actual_probs)[1]
                    
                    accuracies.append(accuracy)
                    confidences.append(confidence)
                    timings.append(end_time - start_time)
            
            mode_results[mode.value] = {
                'mean_accuracy': np.mean(accuracies),
                'mean_confidence': np.mean(confidences),
                'mean_timing': np.mean(timings),
                'std_accuracy': np.std(accuracies)
            }
        
        return mode_results
    
    def _benchmark_hardware_efficiency(self, test_inputs: List[torch.Tensor]) -> Dict:
        """Measure hardware efficiency"""
        
        if not torch.cuda.is_available():
            return {'gpu_available': False}
        
        # Measure GPU utilization and memory usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_memory = torch.cuda.memory_allocated()
        start_time = time.time()
        
        # Run speculation workload
        for input_tensor in test_inputs:
            for layer_id in range(self.speculation_engine.num_layers - 1):
                pred_probs, confidence = self.speculation_engine.predict_next_experts(
                    layer_id, input_tensor
                )
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        return {
            'gpu_available': True,
            'memory_usage_mb': (end_memory - start_memory) / 1024 / 1024,
            'peak_memory_mb': peak_memory / 1024 / 1024,
            'total_time_seconds': end_time - start_time,
            'predictions_per_second': len(test_inputs) * (self.speculation_engine.num_layers - 1) / (end_time - start_time)
        }
    
    def _calculate_top_k_accuracy(self, pred_probs: torch.Tensor, actual_probs: torch.Tensor) -> Dict[int, float]:
        """Calculate top-k accuracy for different k values"""
        
        accuracies = {}
        
        for k in [1, 2, 3]:
            pred_top_k = torch.topk(pred_probs, k=k)[1]
            actual_top_k = torch.topk(actual_probs, k=k)[1]
            
            # Calculate overlap
            overlap = len(set(pred_top_k.tolist()) & set(actual_top_k.tolist()))
            accuracies[k] = overlap / k
        
        return accuracies
    
    def _calculate_probability_correlation(self, pred_probs: torch.Tensor, actual_probs: torch.Tensor) -> float:
        """Calculate correlation between predicted and actual probability distributions"""
        try:
            corr = torch.corrcoef(torch.stack([pred_probs, actual_probs]))[0, 1].item()
            return corr if not torch.isnan(torch.tensor(corr)) else 0.0
        except:
            return 0.0
    
    def _calculate_kl_divergence(self, pred_probs: torch.Tensor, actual_probs: torch.Tensor) -> float:
        """Calculate KL divergence between distributions"""
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        pred_probs = pred_probs + eps
        actual_probs = actual_probs + eps
        
        # Normalize
        pred_probs = pred_probs / pred_probs.sum()
        actual_probs = actual_probs / actual_probs.sum()
        
        return F.kl_div(pred_probs.log(), actual_probs, reduction='sum').item()
    
    def _save_benchmark_results(self, results: Dict):
        """Save benchmark results to file"""
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"speculation_benchmark_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                val = float(obj)
                return val if not np.isnan(val) and not np.isinf(val) else 0.0
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return str(obj)  # For complex objects
            else:
                return obj
        
        serializable_results = convert_for_json(results)
        
        with open(output_dir / filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📊 Benchmark results saved to: {output_dir / filename}")
    
    def generate_report(self, results: Dict) -> str:
        """Generate a human-readable benchmark report"""
        
        report = []
        report.append("🔍 SPECULATION BENCHMARK REPORT")
        report.append("=" * 50)
        
        # Accuracy Summary
        report.append("\n📈 ACCURACY METRICS")
        report.append("-" * 20)
        acc_metrics = results['accuracy']
        report.append(f"Top-1 Accuracy: {acc_metrics['mean_top_k_accuracy'][1]:.3f} ± {acc_metrics['std_top_k_accuracy'][1]:.3f}")
        report.append(f"Top-2 Accuracy: {acc_metrics['mean_top_k_accuracy'][2]:.3f} ± {acc_metrics['std_top_k_accuracy'][2]:.3f}")
        report.append(f"Top-3 Accuracy: {acc_metrics['mean_top_k_accuracy'][3]:.3f} ± {acc_metrics['std_top_k_accuracy'][3]:.3f}")
        report.append(f"Probability Correlation: {acc_metrics['mean_probability_correlation']:.3f}")
        report.append(f"KL Divergence: {acc_metrics['mean_kl_divergence']:.3f}")
        
        # Calibration Summary
        report.append("\n🎯 CONFIDENCE CALIBRATION")
        report.append("-" * 25)
        cal_metrics = results['calibration']
        report.append(f"Expected Calibration Error: {cal_metrics['expected_calibration_error']:.3f}")
        report.append(f"Confidence-Accuracy Correlation: {cal_metrics['confidence_accuracy_correlation']:.3f}")
        
        # Mode Comparison
        report.append("\n🔄 MODE COMPARISON")
        report.append("-" * 17)
        for mode, metrics in results['mode_comparison'].items():
            report.append(f"{mode:15} - Accuracy: {metrics['mean_accuracy']:.3f}, Confidence: {metrics['mean_confidence']:.3f}")
        
        # Efficiency Summary
        report.append("\n⚡ HARDWARE EFFICIENCY")
        report.append("-" * 20)
        eff_metrics = results['efficiency']
        if eff_metrics['gpu_available']:
            report.append(f"Memory Usage: {eff_metrics['memory_usage_mb']:.1f} MB")
            report.append(f"Predictions/sec: {eff_metrics['predictions_per_second']:.1f}")
        else:
            report.append("GPU not available for efficiency testing")
        
        return "\n".join(report)


def create_test_inputs(num_samples: int = 20, batch_size: int = 4, seq_len: int = 128, hidden_size: int = 512) -> List[torch.Tensor]:
    """Generate test inputs with different patterns"""
    
    test_inputs = []
    
    # Generate different types of inputs
    for i in range(num_samples):
        if i % 3 == 0:
            # Repetitive pattern
            base_pattern = torch.randn(1, 1, hidden_size)
            input_tensor = base_pattern.repeat(batch_size, seq_len, 1)
            input_tensor += torch.randn_like(input_tensor) * 0.1  # Small noise
        elif i % 3 == 1:
            # Diverse pattern
            input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        else:
            # Transitional pattern
            input_tensor = torch.randn(batch_size, seq_len, hidden_size)
            input_tensor[:, :seq_len//2] = torch.randn(batch_size, seq_len//2, hidden_size) * 0.5
        
        test_inputs.append(input_tensor)
    
    return test_inputs


if __name__ == "__main__":
    # Example usage
    from ..gating.speculation_engine import create_speculation_engine
    
    print("🚀 Starting Speculation Benchmark")
    
    # Create test engine
    engine = create_speculation_engine(num_experts=8, num_layers=6, mode="multi_layer")
    
    # Generate test inputs
    test_inputs = create_test_inputs(num_samples=10)
    
    # Create benchmark (no model needed for this test)
    benchmark = SpeculationBenchmark(None, engine)
    
    # Run benchmark
    results = benchmark.run_comprehensive_benchmark(test_inputs)
    
    # Generate report
    report = benchmark.generate_report(results)
    print("\n" + report)