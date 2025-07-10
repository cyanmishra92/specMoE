"""
Pre-trained Switch Transformer model integration
Uses Hugging Face Switch Transformers for real MoE experiments
"""

import torch
import torch.nn as nn
from transformers import (
    SwitchTransformersForConditionalGeneration,
    T5Tokenizer,
    SwitchTransformersConfig
)
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)


class PretrainedSwitchWrapper:
    """
    Wrapper for pre-trained Switch Transformer models with speculation support
    """
    
    def __init__(self, model_name: str = "google/switch-base-8"):
        """
        Initialize with a pre-trained Switch Transformer
        
        Available models:
        - google/switch-base-8 (7B params, 8 experts) - Good for RTX 3090
        - google/switch-base-16 (7B params, 16 experts)
        - google/switch-base-32 (7B params, 32 experts)
        - google/switch-base-64 (7B params, 64 experts)
        - google/switch-base-128 (7B params, 128 experts)
        - google/switch-base-256 (7B params, 256 experts)
        """
        self.model_name = model_name
        print(f"Loading pre-trained Switch Transformer: {model_name}")
        
        # Load model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = SwitchTransformersForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use FP16 to save memory
            device_map="auto"
        )
        
        # Get model configuration
        self.config = self.model.config
        
        # Extract MoE information
        self.num_layers = len([m for m in self.model.modules() if hasattr(m, 'mlp')])
        self.num_experts = getattr(self.config, 'num_experts', 8)
        self.expert_capacity = getattr(self.config, 'expert_capacity', 64)
        self.hidden_size = self.config.d_model
        
        # For collecting routing information
        self.routing_history = []
        self.current_routing_info = []
        
        # Hook into the model to collect routing information
        self._setup_routing_hooks()
        
        print(f"Loaded model with {self.get_parameter_count():,} parameters")
        print(f"Experts per layer: {self.num_experts}")
        print(f"Hidden size: {self.hidden_size}")
    
    def _setup_routing_hooks(self):
        """Set up hooks to collect routing information from MoE layers"""
        def routing_hook(module, input, output):
            if hasattr(module, 'router') and hasattr(output, 'router_logits'):
                # Extract routing information
                router_logits = output.router_logits
                if router_logits is not None:
                    routing_info = {
                        'router_logits': router_logits.detach().clone(),
                        'layer_name': str(module),
                        'num_experts': self.num_experts
                    }
                    self.current_routing_info.append(routing_info)
        
        # Register hooks on all MoE layers
        for name, module in self.model.named_modules():
            if 'mlp' in name.lower() and hasattr(module, 'router'):
                module.register_forward_hook(routing_hook)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        collect_routing_stats: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with routing information collection"""
        
        # Clear previous routing info
        self.current_routing_info = []
        
        # Create decoder input ids if not provided (for generation)
        if decoder_input_ids is None:
            # Use pad token as start token
            decoder_input_ids = torch.full(
                (input_ids.shape[0], 1),
                self.tokenizer.pad_token_id,
                device=input_ids.device,
                dtype=input_ids.dtype
            )
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_router_logits=True,
            return_dict=True
        )
        
        # Process routing information
        processed_routing = self._process_routing_info()
        
        # Store routing history
        if collect_routing_stats:
            self.routing_history.append(processed_routing)
            if len(self.routing_history) > 100:  # Keep last 100 entries
                self.routing_history.pop(0)
        
        return {
            'logits': outputs.logits,
            'routing_info': processed_routing,
            'encoder_router_logits': outputs.encoder_router_logits,
            'decoder_router_logits': outputs.decoder_router_logits,
            'speculation_data': self._get_speculation_data()
        }
    
    def _process_routing_info(self) -> List[Dict]:
        """Process raw routing information into standardized format"""
        processed = []
        
        for info in self.current_routing_info:
            router_logits = info['router_logits']
            
            # Compute gate scores (softmax of router logits)
            gate_scores = torch.softmax(router_logits, dim=-1)
            
            # Get top-k experts (typically k=1 for Switch Transformers)
            top_k_gates, top_k_indices = torch.topk(gate_scores, k=1, dim=-1)
            
            # Compute load balancing metrics
            expert_usage = gate_scores.mean(dim=0)  # Average usage per expert
            load_balance_loss = self._compute_load_balancing_loss(gate_scores)
            
            processed_info = {
                'gate_scores': gate_scores,
                'top_k_indices': top_k_indices,
                'top_k_gates': top_k_gates,
                'expert_usage': expert_usage,
                'load_balancing_loss': load_balance_loss,
                'routing_entropy': self._compute_routing_entropy(gate_scores),
                'layer_name': info['layer_name']
            }
            
            processed.append(processed_info)
        
        return processed
    
    def _compute_load_balancing_loss(self, gate_scores: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss for expert utilization"""
        # Fraction of tokens routed to each expert
        router_probs = gate_scores.mean(dim=0)
        
        # Fraction of total router probability allocated to each expert
        expert_mask = torch.nn.functional.one_hot(
            gate_scores.argmax(dim=-1), 
            num_classes=self.num_experts
        ).float()
        expert_usage = expert_mask.mean(dim=0)
        
        # Load balancing loss
        return self.num_experts * torch.sum(router_probs * expert_usage)
    
    def _compute_routing_entropy(self, gate_scores: torch.Tensor) -> torch.Tensor:
        """Compute routing entropy (measure of routing diversity)"""
        return -torch.sum(gate_scores * torch.log(gate_scores + 1e-8), dim=-1).mean()
    
    def _get_speculation_data(self) -> Dict:
        """Get speculation data for the speculation engine"""
        return {
            'routing_history': self.routing_history[-4:],  # Last 4 entries
            'num_experts': self.num_experts,
            'expert_capacity': self.expert_capacity,
            'current_routing': self.current_routing_info
        }
    
    def get_model_info(self) -> Dict:
        """Get model information for device profiling"""
        total_params = self.get_parameter_count()
        
        # Estimate expert parameters (rough approximation)
        # Each expert is typically an MLP with 2 linear layers
        expert_params_per_layer = self.hidden_size * self.hidden_size * 4 * 2  # Approximate
        total_expert_params = expert_params_per_layer * self.num_experts * self.num_layers
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'expert_parameters': total_expert_params,
            'non_expert_parameters': total_params - total_expert_params,
            'num_layers': self.num_layers,
            'num_experts_per_layer': self.num_experts,
            'hidden_size': self.hidden_size,
            'expert_capacity': self.expert_capacity,
            'memory_per_expert_mb': expert_params_per_layer * 2 / (1024 * 1024)  # FP16
        }
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def prepare_inputs(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Prepare text inputs for the model"""
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        return inputs
    
    def generate_with_routing_stats(
        self,
        texts: List[str],
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> Dict:
        """Generate text while collecting routing statistics"""
        
        # Prepare inputs
        inputs = self.prepare_inputs(texts)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_router_logits=True,
                return_dict_in_generate=True,
                **generation_kwargs
            )
        
        # Decode generated text
        generated_texts = self.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True
        )
        
        return {
            'generated_texts': generated_texts,
            'sequences': outputs.sequences,
            'router_logits': outputs.encoder_router_logits if hasattr(outputs, 'encoder_router_logits') else None
        }


def get_available_switch_models() -> List[Dict[str, str]]:
    """Get list of available Switch Transformer models suitable for RTX 3090"""
    return [
        {
            'name': 'google/switch-base-8',
            'description': 'Base model with 8 experts (~7B params) - Recommended for RTX 3090',
            'experts': 8,
            'size': '~7B parameters'
        },
        {
            'name': 'google/switch-base-16',
            'description': 'Base model with 16 experts (~7B params)',
            'experts': 16,
            'size': '~7B parameters'
        },
        {
            'name': 'google/switch-base-32',
            'description': 'Base model with 32 experts (~7B params)',
            'experts': 32,
            'size': '~7B parameters'
        },
        {
            'name': 'google/switch-base-64',
            'description': 'Base model with 64 experts (~7B params) - May need compression',
            'experts': 64,
            'size': '~7B parameters'
        }
    ]


def create_pretrained_switch_model(model_name: str = "google/switch-base-8") -> PretrainedSwitchWrapper:
    """Create a pre-trained Switch Transformer model"""
    return PretrainedSwitchWrapper(model_name)


if __name__ == "__main__":
    # Test loading and basic functionality
    print("Available Switch Transformer models:")
    for model in get_available_switch_models():
        print(f"  - {model['name']}: {model['description']}")
    
    print("\nTesting model loading...")
    try:
        # Test with smallest model first
        model = create_pretrained_switch_model("google/switch-base-8")
        
        # Test basic functionality
        test_texts = ["Translate to French: Hello, how are you?"]
        inputs = model.prepare_inputs(test_texts)
        
        print(f"Input shape: {inputs['input_ids'].shape}")
        print(f"Model info: {model.get_model_info()}")
        
        # Test forward pass
        outputs = model.forward(inputs['input_ids'], inputs['attention_mask'])
        print(f"Output logits shape: {outputs['logits'].shape}")
        print(f"Number of routing layers: {len(outputs['routing_info'])}")
        
        print("✅ Pre-trained Switch Transformer loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("This might be due to memory constraints or missing model files.")