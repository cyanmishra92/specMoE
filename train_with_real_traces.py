#!/usr/bin/env python3
"""
Train Speculation Models with Real MoE Routing Traces
Use collected real routing patterns to train high-accuracy speculation models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import logging
from pathlib import Path
import json
import time
import pickle

# Import our modules
from training.learnable_gating_models import GatingModelConfig, create_gating_model
from training.gating_trainer import TrainingConfig, train_gating_model
from training.gating_data_collector import GatingDataPoint
from evaluation.speculation_benchmark import SpeculationBenchmark

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_real_traces(trace_file="routing_data/real_traces_combined.pkl"):
    """Load real routing traces from file"""
    
    logger.info(f"Loading real traces from {trace_file}")
    
    if not Path(trace_file).exists():
        logger.error(f"Trace file not found: {trace_file}")
        logger.info("Please run 'python collect_real_traces.py' first")
        return None
    
    with open(trace_file, 'rb') as f:
        serializable_traces = pickle.load(f)
    
    # Convert back to GatingDataPoint objects
    traces = []
    for trace_dict in serializable_traces:
        trace = GatingDataPoint(
            layer_id=trace_dict['layer_id'],
            hidden_states=torch.from_numpy(trace_dict['hidden_states']),
            input_embeddings=torch.from_numpy(trace_dict['input_embeddings']),
            target_routing=torch.from_numpy(trace_dict['target_routing']),
            target_top_k=torch.from_numpy(trace_dict['target_top_k']),
            prev_layer_gates=[torch.from_numpy(g) for g in trace_dict['prev_layer_gates']],
            sequence_length=trace_dict['sequence_length'],
            token_ids=torch.from_numpy(trace_dict['token_ids']) if trace_dict['token_ids'] is not None else None,
            dataset_name=trace_dict['dataset_name'],
            sample_id=trace_dict['sample_id']
        )
        traces.append(trace)
    
    logger.info(f"✅ Loaded {len(traces)} real routing traces")
    
    # Print statistics
    layer_counts = {}
    dataset_counts = {}
    for trace in traces:
        layer_counts[trace.layer_id] = layer_counts.get(trace.layer_id, 0) + 1
        dataset_counts[trace.dataset_name] = dataset_counts.get(trace.dataset_name, 0) + 1
    
    logger.info("📊 Trace Statistics:")
    logger.info(f"  Layers: {dict(sorted(layer_counts.items()))}")
    logger.info(f"  Datasets: {dataset_counts}")
    
    return traces

def train_speculation_models_on_real_data(traces):
    """Train speculation models using real routing traces"""
    
    logger.info("🧠 Training Speculation Models on Real Data")
    logger.info("=" * 50)
    
    # Determine model configuration from data
    sample_trace = traces[0]
    num_experts = sample_trace.target_routing.shape[-1]
    max_layer = max(trace.layer_id for trace in traces)
    hidden_size = sample_trace.hidden_states.shape[-1]
    max_seq_len = max(trace.sequence_length for trace in traces)
    
    logger.info(f"Data configuration:")
    logger.info(f"  Num experts: {num_experts}")
    logger.info(f"  Max layer: {max_layer}")
    logger.info(f"  Hidden size: {hidden_size}")
    logger.info(f"  Max sequence length: {max_seq_len}")
    
    # Training configuration optimized for RTX 3090
    training_config = TrainingConfig(
        batch_size=8,           # Larger batch with real GPU
        learning_rate=2e-4,     # Lower LR for better convergence
        num_epochs=20,          # More epochs with real data
        eval_steps=200,
        save_steps=500,
        device="cuda",
        mixed_precision=True,   # Use mixed precision for efficiency
        warmup_steps=500,
        gradient_clip_norm=1.0
    )
    
    gating_config = GatingModelConfig(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_layers=max_layer + 1,
        max_seq_len=min(max_seq_len, 512),  # Cap at 512 for memory
        context_layers=4
    )
    
    # Train different model architectures
    model_types = ["contextual", "transformer", "hierarchical"]
    trained_models = {}
    
    for model_type in model_types:
        logger.info(f"\n📚 Training {model_type} model...")
        
        try:
            start_time = time.time()
            
            model, training_stats = train_gating_model(
                data_points=traces,
                model_type=model_type,
                training_config=training_config,
                gating_config=gating_config
            )
            
            training_time = time.time() - start_time
            
            trained_models[model_type] = {
                'model': model,
                'training_stats': training_stats,
                'training_time': training_time
            }
            
            # Save model
            model_path = f"trained_models/{model_type}_real_data.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'training_stats': training_stats,
                'gating_config': gating_config.__dict__,
                'training_config': training_config.__dict__,
                'num_traces': len(traces)
            }, model_path)
            
            logger.info(f"✅ {model_type} model trained successfully")
            logger.info(f"   Training time: {training_time:.1f}s")
            logger.info(f"   Final train loss: {training_stats['train_losses'][-1]:.4f}")
            if training_stats['val_losses']:
                logger.info(f"   Final val loss: {training_stats['val_losses'][-1]:.4f}")
            logger.info(f"   Model saved to: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to train {model_type} model: {e}")
            continue
    
    return trained_models

def evaluate_trained_models(trained_models, traces):
    """Evaluate trained models and compare with baselines"""
    
    logger.info("📊 Evaluating Trained Models")
    logger.info("=" * 40)
    
    # Create test data
    test_traces = traces[-100:]  # Use last 100 traces for testing
    
    evaluation_results = {}
    
    for model_type, model_data in trained_models.items():
        logger.info(f"\n🔍 Evaluating {model_type} model...")
        
        model = model_data['model']
        
        # Simple evaluation: measure prediction accuracy
        correct_predictions = 0
        total_predictions = 0
        confidence_scores = []
        
        model.eval()
        with torch.no_grad():
            for trace in test_traces[:20]:  # Sample for quick evaluation
                try:
                    # Prepare input
                    hidden_states = trace.hidden_states.unsqueeze(0).cuda()
                    prev_gates = [g.unsqueeze(0).cuda() for g in trace.prev_layer_gates[:3]]
                    
                    if len(prev_gates) == 0:
                        continue
                    
                    # Get prediction
                    gating_logits, confidence, _ = model(
                        hidden_states=hidden_states,
                        prev_layer_gates=prev_gates,
                        layer_id=trace.layer_id
                    )
                    
                    # Calculate accuracy
                    pred_experts = torch.topk(gating_logits.squeeze(0), k=1, dim=-1).indices
                    target_experts = torch.topk(trace.target_routing, k=1, dim=-1).indices
                    
                    # Simple accuracy: fraction of correct predictions per token
                    matches = (pred_experts == target_experts.cuda()).float()
                    accuracy = matches.mean().item()
                    
                    correct_predictions += accuracy
                    total_predictions += 1
                    confidence_scores.append(confidence.mean().item())
                    
                except Exception as e:
                    logger.warning(f"Evaluation error for {model_type}: {e}")
                    continue
        
        if total_predictions > 0:
            avg_accuracy = correct_predictions / total_predictions
            avg_confidence = np.mean(confidence_scores)
            
            evaluation_results[model_type] = {
                'accuracy': avg_accuracy,
                'confidence': avg_confidence,
                'num_predictions': total_predictions
            }
            
            logger.info(f"  Accuracy: {avg_accuracy:.3f}")
            logger.info(f"  Confidence: {avg_confidence:.3f}")
            logger.info(f"  Predictions: {total_predictions}")
        else:
            logger.warning(f"No valid predictions for {model_type}")
    
    return evaluation_results

def create_final_report(trained_models, evaluation_results, traces):
    """Create comprehensive training report"""
    
    report = []
    report.append("🎯 REAL DATA TRAINING RESULTS")
    report.append("=" * 50)
    
    # Data statistics
    report.append(f"\n📊 TRAINING DATA")
    report.append(f"Total traces: {len(traces)}")
    
    layer_counts = {}
    for trace in traces:
        layer_counts[trace.layer_id] = layer_counts.get(trace.layer_id, 0) + 1
    
    report.append("Layer distribution:")
    for layer_id, count in sorted(layer_counts.items()):
        report.append(f"  Layer {layer_id}: {count} traces")
    
    # Training results
    report.append(f"\n🧠 TRAINING RESULTS")
    report.append("-" * 20)
    
    for model_type, model_data in trained_models.items():
        stats = model_data['training_stats']
        report.append(f"\n{model_type.title()} Model:")
        report.append(f"  Training time: {model_data['training_time']:.1f}s")
        report.append(f"  Final train loss: {stats['train_losses'][-1]:.4f}")
        if stats['val_losses']:
            report.append(f"  Final val loss: {stats['val_losses'][-1]:.4f}")
        
        # Show loss improvement
        if len(stats['train_losses']) > 1:
            initial_loss = stats['train_losses'][0]
            final_loss = stats['train_losses'][-1]
            improvement = (initial_loss - final_loss) / initial_loss * 100
            report.append(f"  Loss reduction: {improvement:.1f}%")
    
    # Evaluation results
    if evaluation_results:
        report.append(f"\n📈 EVALUATION RESULTS")
        report.append("-" * 20)
        
        best_model = max(evaluation_results.keys(), 
                        key=lambda x: evaluation_results[x]['accuracy'])
        best_accuracy = evaluation_results[best_model]['accuracy']
        
        for model_type, results in evaluation_results.items():
            marker = "🏆" if model_type == best_model else "  "
            report.append(f"{marker} {model_type:12} - Accuracy: {results['accuracy']:.3f}, Confidence: {results['confidence']:.3f}")
        
        report.append(f"\n🎯 Best model: {best_model} ({best_accuracy:.3f} accuracy)")
    
    # Next steps
    report.append(f"\n🚀 NEXT STEPS")
    report.append("-" * 12)
    report.append("1. Test speculative loading with best model")
    report.append("2. Measure memory savings and speedup")
    report.append("3. Implement multi-layer lookahead (N+1, N+2, N+3)")
    report.append("4. Scale to larger datasets (10,000+ traces)")
    
    report_text = "\n".join(report)
    
    # Save report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"training_reports/real_data_training_{timestamp}.txt"
    Path("training_reports").mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    logger.info(f"📄 Report saved to: {report_file}")
    
    return report_text

def main():
    """Main training pipeline with real data"""
    
    logger.info("🚀 Training Speculation Models with Real MoE Traces")
    logger.info("=" * 60)
    
    # Step 1: Load real traces
    traces = load_real_traces()
    if traces is None or len(traces) == 0:
        logger.error("No traces available. Please run collect_real_traces.py first")
        return False
    
    # Step 2: Train models
    trained_models = train_speculation_models_on_real_data(traces)
    if not trained_models:
        logger.error("No models trained successfully")
        return False
    
    # Step 3: Evaluate models
    evaluation_results = evaluate_trained_models(trained_models, traces)
    
    # Step 4: Generate report
    report = create_final_report(trained_models, evaluation_results, traces)
    print("\n" + report)
    
    logger.info(f"\n✅ Training pipeline completed successfully!")
    logger.info(f"🎯 {len(trained_models)} models trained on {len(traces)} real traces")
    
    if evaluation_results:
        best_model = max(evaluation_results.keys(), 
                        key=lambda x: evaluation_results[x]['accuracy'])
        best_accuracy = evaluation_results[best_model]['accuracy']
        logger.info(f"🏆 Best model: {best_model} with {best_accuracy:.3f} accuracy")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Ready for production speculation!")
        print("Next: Test speculative loading with trained models")
    else:
        print("\n❌ Training failed - check logs for details")