#!/usr/bin/env python3
"""
Complete Gating Model Training Pipeline
1. Collect routing data from real MoE models
2. Train learnable gating models
3. Evaluate and compare with heuristic methods
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import logging
from typing import Dict, List, Tuple, Optional

# Import our modules
from training.gating_data_collector import GatingDataCollector, collect_multiple_datasets
from training.learnable_gating_models import GatingModelConfig, create_gating_model
from training.gating_trainer import TrainingConfig, train_gating_model
from gating.speculation_engine import SpeculationEngine, SpeculationMode
from evaluation.speculation_benchmark import SpeculationBenchmark, create_test_inputs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    dirs = ['routing_data', 'trained_models', 'evaluation_results', 'plots']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

def collect_training_data(force_recollect: bool = False) -> str:
    """
    Collect routing data from benchmark datasets
    """
    logger.info("🔍 Collecting Training Data from Benchmark Datasets")
    logger.info("=" * 60)
    
    data_file = "routing_data/combined_routing_data.pkl"
    
    if Path(data_file).exists() and not force_recollect:
        logger.info(f"Found existing data file: {data_file}")
        return data_file
    
    # Define benchmark datasets to collect from
    datasets_config = [
        {
            'name': 'wikitext', 
            'config': 'wikitext-2-raw-v1', 
            'samples': 500,
            'split': 'train'
        },
        {
            'name': 'squad', 
            'config': 'plain_text', 
            'samples': 300,
            'split': 'train'
        },
        {
            'name': 'glue', 
            'config': 'cola', 
            'samples': 200,
            'split': 'train'
        }
    ]
    
    try:
        # Try with pre-trained Switch Transformer
        logger.info("Attempting to collect data with google/switch-base-8...")
        combined_path = collect_multiple_datasets(
            datasets_config,
            model_name="google/switch-base-8",
            output_dir="routing_data"
        )
        
    except Exception as e:
        logger.warning(f"Failed with pre-trained model: {e}")
        logger.info("Falling back to custom model...")
        
        # Fallback: Use our custom model to generate synthetic routing patterns
        combined_path = collect_synthetic_routing_data(datasets_config)
    
    return combined_path

def collect_synthetic_routing_data(datasets_config: List[Dict]) -> str:
    """
    Generate synthetic routing data using our custom model
    """
    logger.info("Generating synthetic routing data with custom model...")
    
    from models.small_switch_transformer import SmallSwitchTransformer
    from transformers import AutoTokenizer
    
    # Load custom model
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = SmallSwitchTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        num_experts=8
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Generate synthetic data
    from training.gating_data_collector import GatingDataPoint
    
    synthetic_data = []
    
    for dataset_config in datasets_config:
        logger.info(f"Generating data for {dataset_config['name']}...")
        
        for sample_idx in range(dataset_config['samples']):
            # Create synthetic input
            seq_len = np.random.randint(32, 256)
            input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len)).to(device)
            
            # Forward pass to get routing
            with torch.no_grad():
                # Manually forward through layers to collect routing
                hidden_states = model.embeddings(input_ids)
                
                layer_routing_data = []
                
                for layer_idx, layer in enumerate(model.layers):
                    # Forward through attention first
                    attn_output = layer.attention(hidden_states)
                    
                    # Forward through MoE
                    moe_output, routing_info = layer.moe_mlp.forward_with_routing(attn_output)
                    
                    # Store routing data
                    layer_routing_data.append({
                        'layer_id': layer_idx,
                        'hidden_states': attn_output.detach().cpu().squeeze(0),
                        'gate_scores': routing_info['gate_scores'].detach().cpu(),
                        'top_k_indices': routing_info['top_k_indices'].detach().cpu()
                    })
                    
                    hidden_states = moe_output
                
                # Convert to GatingDataPoint objects
                for i, routing_data in enumerate(layer_routing_data):
                    # Get previous layer data for context
                    prev_layer_gates = []
                    for j in range(max(0, i-3), i):
                        if j < len(layer_routing_data):
                            prev_layer_gates.append(layer_routing_data[j]['gate_scores'])
                    
                    data_point = GatingDataPoint(
                        layer_id=routing_data['layer_id'],
                        hidden_states=routing_data['hidden_states'],
                        input_embeddings=input_ids.cpu().squeeze(0),
                        prev_layer_gates=prev_layer_gates,
                        target_routing=routing_data['gate_scores'].squeeze(0),
                        target_top_k=routing_data['top_k_indices'].squeeze(0),
                        sequence_length=seq_len,
                        token_ids=input_ids.cpu().squeeze(0),
                        dataset_name=dataset_config['name'],
                        sample_id=f"synthetic_{sample_idx}"
                    )
                    
                    synthetic_data.append(data_point)
    
    # Save synthetic data
    collector = GatingDataCollector()
    output_path = "routing_data/synthetic_routing_data.pkl"
    collector.save_collected_data(synthetic_data, output_path)
    
    logger.info(f"✅ Generated {len(synthetic_data)} synthetic routing data points")
    
    return output_path

def train_learnable_models(data_file: str) -> Dict[str, Tuple]:
    """
    Train different types of learnable gating models
    """
    logger.info("🧠 Training Learnable Gating Models")
    logger.info("=" * 50)
    
    # Load data
    collector = GatingDataCollector()
    data_points = collector.load_collected_data(data_file)
    
    logger.info(f"Loaded {len(data_points)} training data points")
    
    # Training configurations
    training_config = TrainingConfig(
        batch_size=8,  # Small batch for RTX 3090
        learning_rate=5e-4,
        num_epochs=20,
        eval_steps=200,
        save_steps=500,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    gating_config = GatingModelConfig(
        hidden_size=512,
        num_experts=8,
        num_layers=6,
        max_seq_len=256  # Smaller for memory efficiency
    )
    
    # Train different model types
    model_types = ["contextual", "transformer", "hierarchical"]
    trained_models = {}
    
    for model_type in model_types:
        logger.info(f"\n📚 Training {model_type} model...")
        
        try:
            model, training_stats = train_gating_model(
                data_points=data_points,
                model_type=model_type,
                training_config=training_config,
                gating_config=gating_config
            )
            
            trained_models[model_type] = (model, training_stats)
            
            # Save model
            model_path = f"trained_models/{model_type}_gating_model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'training_stats': training_stats,
                'config': gating_config
            }, model_path)
            
            logger.info(f"✅ {model_type} model trained successfully")
            logger.info(f"   Final train loss: {training_stats['train_losses'][-1]:.4f}")
            if training_stats['val_losses']:
                logger.info(f"   Final val loss: {training_stats['val_losses'][-1]:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to train {model_type} model: {e}")
            continue
    
    return trained_models

def evaluate_all_models(trained_models: Dict[str, Tuple]) -> Dict:
    """
    Comprehensive evaluation of all models (heuristic + learnable)
    """
    logger.info("📊 Evaluating All Models")
    logger.info("=" * 30)
    
    # Generate test inputs
    test_inputs = create_test_inputs(num_samples=100, batch_size=2, seq_len=128)
    
    evaluation_results = {}
    
    # 1. Evaluate heuristic methods (baseline)
    logger.info("Evaluating heuristic methods...")
    
    heuristic_engines = {
        "layer_minus_1": SpeculationEngine(num_experts=8, num_layers=6, 
                                          speculation_mode=SpeculationMode.LAYER_MINUS_1),
        "multi_layer": SpeculationEngine(num_experts=8, num_layers=6, 
                                        speculation_mode=SpeculationMode.MULTI_LAYER),
        "pattern": SpeculationEngine(num_experts=8, num_layers=6, 
                                    speculation_mode=SpeculationMode.PATTERN_LEARNING),
        "adaptive": SpeculationEngine(num_experts=8, num_layers=6, 
                                     speculation_mode=SpeculationMode.ADAPTIVE)
    }
    
    for name, engine in heuristic_engines.items():
        logger.info(f"  Testing {name}...")
        benchmark = SpeculationBenchmark(None, engine)
        results = benchmark.run_comprehensive_benchmark(test_inputs, save_results=False)
        evaluation_results[f"heuristic_{name}"] = results
    
    # 2. Evaluate learnable models
    logger.info("Evaluating learnable models...")
    
    for model_type, (model, training_stats) in trained_models.items():
        logger.info(f"  Testing {model_type}...")
        
        # Create learnable speculation engine
        learnable_engine = LearnableSpeculationEngine(model, num_experts=8, num_layers=6)
        
        benchmark = SpeculationBenchmark(None, learnable_engine)
        results = benchmark.run_comprehensive_benchmark(test_inputs, save_results=False)
        evaluation_results[f"learnable_{model_type}"] = results
    
    return evaluation_results

class LearnableSpeculationEngine:
    """
    Wrapper to make learnable models compatible with SpeculationBenchmark
    """
    
    def __init__(self, model, num_experts: int, num_layers: int):
        self.model = model
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.model.eval()
        
        # Storage for routing history
        self.prev_layer_gates = []
        
    def predict_next_experts(self, current_layer: int, hidden_states: torch.Tensor, 
                            current_routing: Optional[Dict] = None) -> Tuple[torch.Tensor, float]:
        """Predict expert usage using the learnable model"""
        
        if current_layer >= self.num_layers - 1:
            return torch.ones(self.num_experts) / self.num_experts, 0.0
        
        # Prepare input for the model
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Ensure we have some previous layer data
        if len(self.prev_layer_gates) == 0:
            # Initialize with uniform distribution
            uniform_gates = torch.ones(batch_size, seq_len, self.num_experts, device=hidden_states.device) / self.num_experts
            self.prev_layer_gates.append(uniform_gates)
        
        # Move previous gates to correct device
        prev_gates_device = []
        for gates in self.prev_layer_gates[-3:]:
            prev_gates_device.append(gates.to(hidden_states.device))
        
        with torch.no_grad():
            # Get prediction from learnable model
            gating_logits, confidence, _ = self.model(
                hidden_states=hidden_states,
                prev_layer_gates=prev_gates_device,  # Last 3 layers
                layer_id=current_layer + 1  # Predicting next layer
            )
            
            # Convert to probabilities
            gating_probs = torch.softmax(gating_logits, dim=-1)
            
            # Average over batch and sequence dimensions
            avg_probs = torch.mean(gating_probs, dim=(0, 1))
            avg_confidence = torch.mean(confidence).item()
            
            return avg_probs, avg_confidence
    
    def update_routing_history(self, layer_id: int, routing_info: Dict):
        """Update routing history"""
        gate_scores = routing_info['gate_scores']
        
        # Store for next prediction
        self.prev_layer_gates.append(gate_scores)
        
        # Keep only recent history
        if len(self.prev_layer_gates) > 4:
            self.prev_layer_gates.pop(0)
    
    def update_accuracy(self, layer_id: int, predicted_experts: torch.Tensor, actual_experts: torch.Tensor):
        """Update accuracy (stub for compatibility)"""
        pass
    
    def get_statistics(self) -> Dict:
        """Get statistics (stub for compatibility)"""
        return {
            'overall_accuracy': 0.0,
            'total_predictions': 0,
            'layer_accuracies': {},
            'speculation_mode': 'learnable',
            'confidence_threshold': 0.7,
            'avg_confidence': 0.0
        }
    
    def reset_statistics(self):
        """Reset statistics"""
        self.prev_layer_gates.clear()

def create_comparison_report(evaluation_results: Dict) -> str:
    """Create a comprehensive comparison report"""
    
    logger.info("📈 Creating Comparison Report")
    
    # Extract key metrics
    report_data = {}
    
    for method_name, results in evaluation_results.items():
        acc_results = results['accuracy']
        cal_results = results['calibration']
        
        report_data[method_name] = {
            'top1_accuracy': acc_results['mean_top_k_accuracy'][1],
            'top2_accuracy': acc_results['mean_top_k_accuracy'][2], 
            'kl_divergence': acc_results['mean_kl_divergence'],
            'confidence_correlation': cal_results['confidence_accuracy_correlation'],
            'calibration_error': cal_results['expected_calibration_error']
        }
    
    # Create comparison plots
    create_comparison_plots(report_data)
    
    # Generate text report
    report = []
    report.append("🎯 GATING MODEL COMPARISON REPORT")
    report.append("=" * 50)
    
    # Separate heuristic and learnable methods
    heuristic_methods = {k: v for k, v in report_data.items() if k.startswith('heuristic_')}
    learnable_methods = {k: v for k, v in report_data.items() if k.startswith('learnable_')}
    
    report.append("\n📊 HEURISTIC METHODS")
    report.append("-" * 25)
    for method, metrics in heuristic_methods.items():
        name = method.replace('heuristic_', '').title()
        report.append(f"{name:15} - Top-1: {metrics['top1_accuracy']:.3f}, Top-2: {metrics['top2_accuracy']:.3f}, KL: {metrics['kl_divergence']:.3f}")
    
    report.append("\n🧠 LEARNABLE METHODS")
    report.append("-" * 23)
    for method, metrics in learnable_methods.items():
        name = method.replace('learnable_', '').title()
        report.append(f"{name:15} - Top-1: {metrics['top1_accuracy']:.3f}, Top-2: {metrics['top2_accuracy']:.3f}, KL: {metrics['kl_divergence']:.3f}")
    
    # Find best methods
    all_methods = {**heuristic_methods, **learnable_methods}
    best_method = max(all_methods.keys(), key=lambda x: all_methods[x]['top1_accuracy'])
    best_accuracy = all_methods[best_method]['top1_accuracy']
    
    report.append(f"\n🏆 BEST METHOD")
    report.append("-" * 15)
    report.append(f"Method: {best_method}")
    report.append(f"Top-1 Accuracy: {best_accuracy:.3f}")
    
    # Improvement analysis
    if learnable_methods:
        best_heuristic_acc = max(heuristic_methods.values(), key=lambda x: x['top1_accuracy'])['top1_accuracy']
        best_learnable_acc = max(learnable_methods.values(), key=lambda x: x['top1_accuracy'])['top1_accuracy']
        
        improvement = best_learnable_acc - best_heuristic_acc
        improvement_pct = (improvement / best_heuristic_acc) * 100
        
        report.append(f"\n💡 LEARNABLE vs HEURISTIC")
        report.append("-" * 30)
        report.append(f"Best heuristic accuracy: {best_heuristic_acc:.3f}")
        report.append(f"Best learnable accuracy: {best_learnable_acc:.3f}")
        report.append(f"Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")
    
    return "\n".join(report)

def create_comparison_plots(report_data: Dict):
    """Create visualization plots"""
    
    logger.info("Creating comparison plots...")
    
    methods = list(report_data.keys())
    top1_accs = [report_data[m]['top1_accuracy'] for m in methods]
    top2_accs = [report_data[m]['top2_accuracy'] for m in methods]
    kl_divs = [report_data[m]['kl_divergence'] for m in methods]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Gating Model Comparison: Heuristic vs Learnable', fontsize=16)
    
    # Top-1 Accuracy
    axes[0, 0].bar(range(len(methods)), top1_accs, alpha=0.7)
    axes[0, 0].set_title('Top-1 Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(range(len(methods)))
    axes[0, 0].set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right')
    
    # Top-2 Accuracy  
    axes[0, 1].bar(range(len(methods)), top2_accs, alpha=0.7, color='orange')
    axes[0, 1].set_title('Top-2 Accuracy')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_xticks(range(len(methods)))
    axes[0, 1].set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right')
    
    # KL Divergence (lower is better)
    axes[1, 0].bar(range(len(methods)), kl_divs, alpha=0.7, color='red')
    axes[1, 0].set_title('KL Divergence (Lower is Better)')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_xticks(range(len(methods)))
    axes[1, 0].set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right')
    
    # Combined score (accuracy - normalized_kl)
    max_kl = max(kl_divs)
    normalized_kl = [kl / max_kl for kl in kl_divs]
    combined_scores = [acc - nkl for acc, nkl in zip(top1_accs, normalized_kl)]
    
    axes[1, 1].bar(range(len(methods)), combined_scores, alpha=0.7, color='purple')
    axes[1, 1].set_title('Combined Score (Accuracy - Normalized KL)')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xticks(range(len(methods)))
    axes[1, 1].set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('plots/gating_model_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("📊 Comparison plots saved to plots/gating_model_comparison.png")

def main():
    """Main training and evaluation pipeline"""
    
    logger.info("🚀 Starting Comprehensive Gating Model Training & Evaluation")
    logger.info("=" * 70)
    
    # Setup
    setup_directories()
    
    # Step 1: Collect training data
    logger.info("\n" + "="*70)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("="*70)
    
    data_file = collect_training_data(force_recollect=False)
    
    # Step 2: Train learnable models
    logger.info("\n" + "="*70)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("="*70)
    
    trained_models = train_learnable_models(data_file)
    
    if not trained_models:
        logger.error("❌ No models were trained successfully!")
        return
    
    # Step 3: Comprehensive evaluation
    logger.info("\n" + "="*70)
    logger.info("STEP 3: EVALUATION")
    logger.info("="*70)
    
    evaluation_results = evaluate_all_models(trained_models)
    
    # Step 4: Generate report
    logger.info("\n" + "="*70)
    logger.info("STEP 4: ANALYSIS")
    logger.info("="*70)
    
    report = create_comparison_report(evaluation_results)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save evaluation results
    results_file = f"evaluation_results/gating_evaluation_{timestamp}.json"
    with open(results_file, 'w') as f:
        # Convert complex objects to strings for JSON serialization
        serializable_results = {}
        for key, value in evaluation_results.items():
            serializable_results[key] = str(value)  # Simple string conversion
        json.dump(serializable_results, f, indent=2)
    
    # Save report
    report_file = f"evaluation_results/comparison_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Print final report
    print("\n" + report)
    
    logger.info(f"\n✅ Complete pipeline finished successfully!")
    logger.info(f"📁 Results saved to:")
    logger.info(f"   - {results_file}")
    logger.info(f"   - {report_file}")
    logger.info(f"   - plots/gating_model_comparison.png")

if __name__ == "__main__":
    main()