"""
Enhanced Speculation Engine for MoE Expert Selection
Implements multi-layer lookahead and confidence-based speculation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import math
import numpy as np
from collections import deque, defaultdict


class SpeculationMode(Enum):
    """Different speculation strategies"""
    NONE = "none"                    # No speculation
    LAYER_MINUS_1 = "layer_minus_1"  # Use previous layer only
    MULTI_LAYER = "multi_layer"      # Use multiple previous layers
    PATTERN_LEARNING = "pattern"     # Learn expert patterns
    ADAPTIVE = "adaptive"            # Adapt based on confidence
    LEARNABLE = "learnable"          # Use trained neural models


class InputType(Enum):
    """Input classification for adaptive speculation"""
    REPETITIVE = "repetitive"    # Low entropy, stable patterns
    DIVERSE = "diverse"          # High entropy, unpredictable
    TRANSITIONAL = "transitional" # Medium entropy


class SpeculationEngine:
    """
    Core speculation engine that predicts expert usage for future layers
    """
    
    def __init__(
        self,
        num_experts: int = 8,
        num_layers: int = 6,
        speculation_mode: SpeculationMode = SpeculationMode.MULTI_LAYER,
        confidence_threshold: float = 0.7,
        history_length: int = 4
    ):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.speculation_mode = speculation_mode
        self.confidence_threshold = confidence_threshold
        self.history_length = history_length
        
        # Layer-wise routing history
        self.routing_history = defaultdict(lambda: deque(maxlen=history_length))
        self.gate_scores_history = defaultdict(lambda: deque(maxlen=history_length))
        
        # Speculation accuracy tracking
        self.speculation_accuracy = defaultdict(lambda: deque(maxlen=100))
        self.layer_accuracies = defaultdict(float)
        
        # Expert usage patterns
        self.expert_transition_matrix = torch.zeros(num_experts, num_experts)
        self.expert_usage_history = deque(maxlen=1000)
        
        # Adaptive parameters
        self.speculation_aggressiveness = 0.7
        self.lookahead_layers = 2
        
        # Performance metrics
        self.metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'speculation_hits': 0,
            'speculation_misses': 0,
            'confidence_scores': deque(maxlen=1000)
        }
    
    def update_routing_history(self, layer_id: int, routing_info: Dict):
        """Update routing history for a layer"""
        top_k_indices = routing_info['top_k_indices']
        gate_scores = routing_info['gate_scores']
        
        self.routing_history[layer_id].append(top_k_indices.detach().clone())
        self.gate_scores_history[layer_id].append(gate_scores.detach().clone())
        
        # Update expert transition patterns
        self._update_expert_transitions(top_k_indices)
    
    def _update_expert_transitions(self, expert_indices: torch.Tensor):
        """Update expert transition matrix for pattern learning"""
        # Flatten to get expert usage for this step
        experts_used = expert_indices.flatten().unique()
        
        if len(self.expert_usage_history) > 0:
            prev_experts = self.expert_usage_history[-1]
            # Update transition probabilities
            for prev_expert in prev_experts:
                for curr_expert in experts_used:
                    self.expert_transition_matrix[prev_expert, curr_expert] += 1
        
        self.expert_usage_history.append(experts_used)
    
    def analyze_input_type(self, hidden_states: torch.Tensor) -> InputType:
        """Analyze input characteristics for adaptive speculation"""
        # Compute attention entropy as a proxy for input complexity
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Simple entropy calculation based on hidden state variance
        token_variance = torch.var(hidden_states, dim=-1)  # [batch_size, seq_len]
        mean_variance = torch.mean(token_variance)
        
        # Thresholds determined empirically (would be tuned)
        if mean_variance < 0.1:
            return InputType.REPETITIVE
        elif mean_variance > 0.5:
            return InputType.DIVERSE
        else:
            return InputType.TRANSITIONAL
    
    def compute_speculation_confidence(self, layer_id: int, gate_scores: torch.Tensor) -> float:
        """Compute confidence score for speculation"""
        if len(self.gate_scores_history[layer_id]) == 0:
            return 0.5  # Default medium confidence
        
        # Confidence based on gate score distribution
        max_scores = torch.max(gate_scores, dim=-1)[0]
        entropy = -torch.sum(gate_scores * torch.log(gate_scores + 1e-8), dim=-1)
        
        # High max scores and low entropy indicate confident routing
        confidence = torch.mean(max_scores) / (1.0 + torch.mean(entropy))
        
        # Factor in historical accuracy
        historical_accuracy = self.layer_accuracies.get(layer_id, 0.5)
        final_confidence = 0.7 * confidence.item() + 0.3 * historical_accuracy
        
        return min(1.0, max(0.0, final_confidence))
    
    def should_speculate(self, layer_id: int, gate_scores: torch.Tensor, input_type: InputType) -> bool:
        """Decide whether to perform speculation for this layer"""
        if self.speculation_mode == SpeculationMode.NONE:
            return False
        
        confidence = self.compute_speculation_confidence(layer_id, gate_scores)
        
        # Adapt threshold based on input type
        if input_type == InputType.REPETITIVE:
            threshold = self.confidence_threshold * 0.8  # Lower threshold
        elif input_type == InputType.DIVERSE:
            threshold = self.confidence_threshold * 1.2  # Higher threshold
        else:
            threshold = self.confidence_threshold
        
        return confidence > threshold
    
    def predict_next_experts(
        self, 
        current_layer: int, 
        hidden_states: torch.Tensor,
        current_routing: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Predict expert usage for the next layer
        Returns: (predicted_expert_probs, confidence)
        """
        if self.speculation_mode == SpeculationMode.NONE:
            return torch.ones(self.num_experts) / self.num_experts, 0.0
        
        next_layer = current_layer + 1
        if next_layer >= self.num_layers:
            return torch.ones(self.num_experts) / self.num_experts, 0.0
        
        # Analyze input type
        input_type = self.analyze_input_type(hidden_states)
        
        if current_routing and not self.should_speculate(current_layer, current_routing['gate_scores'], input_type):
            return torch.ones(self.num_experts) / self.num_experts, 0.0
        
        if self.speculation_mode == SpeculationMode.LAYER_MINUS_1:
            return self._predict_layer_minus_1(next_layer)
        elif self.speculation_mode == SpeculationMode.MULTI_LAYER:
            return self._predict_multi_layer(next_layer, input_type)
        elif self.speculation_mode == SpeculationMode.PATTERN_LEARNING:
            return self._predict_pattern_based(next_layer)
        elif self.speculation_mode == SpeculationMode.ADAPTIVE:
            return self._predict_adaptive(next_layer, input_type, hidden_states)
        
        return torch.ones(self.num_experts) / self.num_experts, 0.0
    
    def _predict_layer_minus_1(self, target_layer: int) -> Tuple[torch.Tensor, float]:
        """Simple prediction using previous layer's routing"""
        prev_layer = target_layer - 1
        
        if len(self.gate_scores_history[prev_layer]) == 0:
            return torch.ones(self.num_experts) / self.num_experts, 0.0
        
        # Use the most recent gate scores from previous layer
        prev_gate_scores = self.gate_scores_history[prev_layer][-1]
        predicted_probs = torch.mean(prev_gate_scores, dim=0).cpu()  # Average over batch and move to CPU
        
        confidence = self.compute_speculation_confidence(prev_layer, prev_gate_scores)
        return predicted_probs, confidence
    
    def _predict_multi_layer(self, target_layer: int, input_type: InputType) -> Tuple[torch.Tensor, float]:
        """Multi-layer weighted prediction"""
        # Weights for different layers (more recent = higher weight)
        weights = [0.1, 0.2, 0.3, 0.4]  # L-4, L-3, L-2, L-1
        predicted_probs = torch.zeros(self.num_experts)
        total_weight = 0.0
        total_confidence = 0.0
        
        for i, weight in enumerate(reversed(weights)):
            layer_id = target_layer - 1 - i
            if layer_id < 0 or len(self.gate_scores_history[layer_id]) == 0:
                continue
            
            gate_scores = self.gate_scores_history[layer_id][-1]
            # Aggregate over all dimensions except the last (num_experts)
            if gate_scores.dim() > 2:
                layer_probs = torch.mean(gate_scores.view(-1, gate_scores.size(-1)), dim=0).cpu()
            else:
                layer_probs = torch.mean(gate_scores, dim=0).cpu()
            
            # Adjust weight based on input type
            if input_type == InputType.REPETITIVE:
                adjusted_weight = weight * 1.2  # Trust history more
            elif input_type == InputType.DIVERSE:
                adjusted_weight = weight * 0.8  # Trust history less
            else:
                adjusted_weight = weight
            
            predicted_probs += adjusted_weight * layer_probs
            total_weight += adjusted_weight
            
            confidence = self.compute_speculation_confidence(layer_id, gate_scores)
            total_confidence += adjusted_weight * confidence
        
        if total_weight > 0:
            predicted_probs /= total_weight
            total_confidence /= total_weight
        else:
            predicted_probs = torch.ones(self.num_experts) / self.num_experts
            total_confidence = 0.0
        
        return predicted_probs, total_confidence
    
    def _predict_pattern_based(self, target_layer: int) -> Tuple[torch.Tensor, float]:
        """Pattern-based prediction using expert transition matrix"""
        if len(self.expert_usage_history) == 0:
            return torch.ones(self.num_experts) / self.num_experts, 0.0
        
        # Get recently used experts
        recent_experts = self.expert_usage_history[-1]
        
        # Predict next experts based on transition matrix
        predicted_probs = torch.zeros(self.num_experts)
        for expert in recent_experts:
            # Normalize transition probabilities
            transitions = self.expert_transition_matrix[expert].cpu()
            if transitions.sum() > 0:
                transitions = transitions / transitions.sum()
                predicted_probs += transitions
        
        if len(recent_experts) > 0:
            predicted_probs /= len(recent_experts)
        else:
            predicted_probs = torch.ones(self.num_experts) / self.num_experts
        
        # Confidence based on how much data we have
        total_transitions = self.expert_transition_matrix.sum()
        confidence = min(1.0, total_transitions.item() / 1000.0)  # More data = higher confidence
        
        return predicted_probs, confidence
    
    def _predict_adaptive(
        self, 
        target_layer: int, 
        input_type: InputType, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Adaptive prediction combining multiple strategies"""
        # Combine multi-layer and pattern-based predictions
        multi_layer_probs, multi_layer_conf = self._predict_multi_layer(target_layer, input_type)
        pattern_probs, pattern_conf = self._predict_pattern_based(target_layer)
        
        # Weight combination based on confidence and input type
        if input_type == InputType.REPETITIVE:
            # Trust patterns more for repetitive input
            alpha = 0.3 * multi_layer_conf + 0.7 * pattern_conf
        elif input_type == InputType.DIVERSE:
            # Trust recent history more for diverse input
            alpha = 0.7 * multi_layer_conf + 0.3 * pattern_conf
        else:
            # Balanced combination
            alpha = 0.5 * multi_layer_conf + 0.5 * pattern_conf
        
        # Normalize alpha
        if alpha > 0:
            w1 = multi_layer_conf / alpha
            w2 = pattern_conf / alpha
        else:
            w1 = w2 = 0.5
        
        combined_probs = w1 * multi_layer_probs + w2 * pattern_probs
        combined_confidence = alpha
        
        return combined_probs, combined_confidence
    
    def update_accuracy(self, layer_id: int, predicted_experts: torch.Tensor, actual_experts: torch.Tensor):
        """Update speculation accuracy statistics"""
        # Compute overlap between predicted and actual top experts
        pred_top_k = torch.topk(predicted_experts, k=2)[1]
        # Handle multi-dimensional actual_experts by flattening first
        if actual_experts.dim() > 1:
            actual_flat = actual_experts.view(-1, actual_experts.size(-1)).mean(dim=0)
        else:
            actual_flat = actual_experts
        actual_top_k = torch.topk(actual_flat, k=2)[1]
        
        # Flatten tensors before converting to lists
        overlap = len(set(pred_top_k.flatten().tolist()) & set(actual_top_k.flatten().tolist()))
        accuracy = overlap / len(pred_top_k)
        
        self.speculation_accuracy[layer_id].append(accuracy)
        
        # Update running average
        if len(self.speculation_accuracy[layer_id]) > 0:
            self.layer_accuracies[layer_id] = np.mean(list(self.speculation_accuracy[layer_id]))
        
        # Update global metrics
        self.metrics['total_predictions'] += 1
        if accuracy > 0.5:  # Consider it a hit if >50% overlap
            self.metrics['correct_predictions'] += 1
    
    def get_statistics(self) -> Dict:
        """Get speculation engine statistics"""
        overall_accuracy = 0.0
        if self.metrics['total_predictions'] > 0:
            overall_accuracy = self.metrics['correct_predictions'] / self.metrics['total_predictions']
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_predictions': self.metrics['total_predictions'],
            'layer_accuracies': dict(self.layer_accuracies),
            'speculation_mode': self.speculation_mode.value,
            'confidence_threshold': self.confidence_threshold,
            'avg_confidence': np.mean(list(self.metrics['confidence_scores'])) if self.metrics['confidence_scores'] else 0.0
        }
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.speculation_accuracy.clear()
        self.layer_accuracies.clear()
        self.metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'speculation_hits': 0,
            'speculation_misses': 0,
            'confidence_scores': deque(maxlen=1000)
        }


class SpeculativeGatingWrapper:
    """
    Wrapper that integrates speculation engine with MoE model
    """
    
    def __init__(self, model, speculation_engine: SpeculationEngine):
        self.model = model
        self.speculation_engine = speculation_engine
        self.predictions = {}  # Store predictions for each layer
        
    def forward(self, *args, **kwargs):
        """Forward pass with speculation"""
        # Store original forward method
        original_forwards = {}
        
        for layer_idx, layer in enumerate(self.model.layers):
            # Wrap each layer's forward method
            original_forwards[layer_idx] = layer.moe_mlp.forward
            layer.moe_mlp.forward = self._create_speculative_forward(
                layer_idx, 
                layer.moe_mlp, 
                original_forwards[layer_idx]
            )
        
        try:
            # Run forward pass
            outputs = self.model(*args, **kwargs)
            
            # Add speculation statistics
            outputs['speculation_stats'] = self.speculation_engine.get_statistics()
            
            return outputs
        finally:
            # Restore original forward methods
            for layer_idx, layer in enumerate(self.model.layers):
                layer.moe_mlp.forward = original_forwards[layer_idx]
    
    def _create_speculative_forward(self, layer_idx, moe_layer, original_forward):
        """Create a speculative forward function for a layer"""
        def speculative_forward(x):
            # Get prediction for next layer (if not last layer)
            if layer_idx < len(self.model.layers) - 1:
                pred_probs, confidence = self.speculation_engine.predict_next_experts(
                    layer_idx, x
                )
                self.predictions[layer_idx + 1] = {
                    'predicted_probs': pred_probs,
                    'confidence': confidence
                }
            
            # Run original forward
            output, routing_info = original_forward(x)
            
            # Update speculation engine with actual routing
            self.speculation_engine.update_routing_history(layer_idx, routing_info)
            
            # Check accuracy of previous prediction
            if layer_idx in self.predictions:
                pred_data = self.predictions[layer_idx]
                self.speculation_engine.update_accuracy(
                    layer_idx,
                    pred_data['predicted_probs'],
                    routing_info['gate_scores']
                )
            
            return output, routing_info
        
        return speculative_forward


class LearnableSpeculationEngine:
    """
    Speculation engine that uses trained neural models for prediction
    """
    
    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        gating_model: Optional[nn.Module] = None,
        model_path: Optional[str] = None
    ):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.gating_model = gating_model
        
        # Storage for routing history  
        self.prev_layer_gates = []
        self.max_history = 4
        
        # Load model if path provided
        if model_path and not gating_model:
            self._load_model(model_path)
        
        # Set model to eval mode
        if self.gating_model:
            self.gating_model.eval()
    
    def _load_model(self, model_path: str):
        """Load trained gating model"""
        try:
            from training.learnable_gating_models import create_gating_model, GatingModelConfig
            
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            config_dict = checkpoint.get('gating_config', {})
            
            # Create config
            config = GatingModelConfig(**config_dict)
            
            # Determine model type from path
            if 'contextual' in model_path:
                model_type = 'contextual'
            elif 'transformer' in model_path:
                model_type = 'transformer'
            elif 'hierarchical' in model_path:
                model_type = 'hierarchical'
            else:
                model_type = 'contextual'
            
            # Create and load model
            self.gating_model = create_gating_model(model_type, config)
            self.gating_model.load_state_dict(checkpoint['model_state_dict'])
            self.gating_model.eval()
            
        except Exception as e:
            print(f"Warning: Failed to load model from {model_path}: {e}")
            self.gating_model = None
    
    def predict_next_experts(
        self,
        current_layer: int,
        hidden_states: torch.Tensor,
        current_routing: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """Predict expert usage using the learnable model"""
        
        if not self.gating_model or current_layer >= self.num_layers - 1:
            # Fallback to uniform distribution
            uniform_probs = torch.ones(self.num_experts) / self.num_experts
            return uniform_probs, 0.0
        
        try:
            # Store current routing in history
            if current_routing is not None:
                self.prev_layer_gates.append(current_routing)
                if len(self.prev_layer_gates) > self.max_history:
                    self.prev_layer_gates.pop(0)
            
            # Prepare inputs
            batch_size, seq_len = hidden_states.shape[:2]
            
            # Get previous layer gates (pad if needed)
            prev_gates = []
            for gate in self.prev_layer_gates[-3:]:  # Use last 3 layers
                if gate.dim() == 2:
                    gate = gate.unsqueeze(0).expand(batch_size, -1, -1)
                prev_gates.append(gate)
            
            # Pad with zeros if insufficient history
            while len(prev_gates) < 3:
                zero_gate = torch.zeros(batch_size, seq_len, self.num_experts)
                prev_gates.insert(0, zero_gate)
            
            # Run model prediction
            with torch.no_grad():
                gating_logits, confidence, _ = self.gating_model(
                    hidden_states=hidden_states,
                    prev_layer_gates=prev_gates,
                    layer_id=current_layer + 1
                )
                
                # Convert to probabilities
                gating_probs = F.softmax(gating_logits, dim=-1)
                
                # Average over sequence and batch dimensions
                avg_probs = gating_probs.mean(dim=(0, 1))
                avg_confidence = confidence.mean().item()
                
                return avg_probs, avg_confidence
        
        except Exception as e:
            print(f"Warning: Learnable prediction failed: {e}")
            # Fallback to uniform
            uniform_probs = torch.ones(self.num_experts) / self.num_experts
            return uniform_probs, 0.0
    
    def update_routing_history(self, layer_id: int, routing_info: Dict):
        """Update routing history (for compatibility)"""
        if 'gate_scores' in routing_info:
            gate_scores = routing_info['gate_scores']
            # Convert to batch format if needed
            if gate_scores.dim() == 2:
                gate_scores = gate_scores.view(1, -1, gate_scores.shape[-1])
            self.prev_layer_gates.append(gate_scores)
            
            if len(self.prev_layer_gates) > self.max_history:
                self.prev_layer_gates.pop(0)


def create_speculation_engine(
    num_experts: int = 8,
    num_layers: int = 6,
    mode: str = "multi_layer"
) -> Union[SpeculationEngine, LearnableSpeculationEngine]:
    """Factory function to create speculation engine"""
    
    # Check for learnable mode
    if mode == "learnable":
        # Try to find a trained model
        from pathlib import Path
        model_dir = Path("trained_models")
        
        if model_dir.exists():
            # Look for trained models (prefer real data models)
            model_files = list(model_dir.glob("*real_data.pt"))
            if not model_files:
                model_files = list(model_dir.glob("*.pt"))
            
            if model_files:
                # Use the first available model
                model_path = str(model_files[0])
                return LearnableSpeculationEngine(
                    num_experts=num_experts,
                    num_layers=num_layers,
                    model_path=model_path
                )
        
        # Fallback to regular speculation if no model found
        print("Warning: No trained models found for learnable mode, falling back to multi_layer")
        mode = "multi_layer"
    
    # Create regular speculation engine
    speculation_mode = SpeculationMode(mode)
    
    return SpeculationEngine(
        num_experts=num_experts,
        num_layers=num_layers,
        speculation_mode=speculation_mode,
        confidence_threshold=0.7,
        history_length=4
    )


if __name__ == "__main__":
    # Test speculation engine
    engine = create_speculation_engine()
    
    # Simulate some routing data
    batch_size, seq_len, num_experts = 2, 128, 8
    
    for layer_id in range(6):
        # Simulate gate scores
        gate_scores = F.softmax(torch.randn(batch_size * seq_len, num_experts), dim=-1)
        top_k_indices = torch.topk(gate_scores, k=1)[1]
        
        routing_info = {
            'gate_scores': gate_scores,
            'top_k_indices': top_k_indices
        }
        
        engine.update_routing_history(layer_id, routing_info)
        
        # Test prediction
        hidden_states = torch.randn(batch_size, seq_len, 512)
        pred_probs, confidence = engine.predict_next_experts(layer_id, hidden_states, routing_info)
        
        print(f"Layer {layer_id}: Predicted probs shape: {pred_probs.shape}, Confidence: {confidence:.3f}")
    
    print(f"Final statistics: {engine.get_statistics()}")