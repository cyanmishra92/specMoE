"""
Adaptive Memory Manager for MoE Expert Loading
Implements hierarchical caching and dynamic buffer management
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import threading
import time
from collections import OrderedDict, defaultdict
import numpy as np
import psutil


class BufferStrategy(Enum):
    """Different buffering strategies based on available memory"""
    DOUBLE_BUFFER = "double_buffer"      # High memory: full double buffering
    SINGLE_BUFFER_ASYNC = "single_async" # Medium memory: single buffer with async loading
    STREAMING = "streaming"              # Low memory: stream experts as needed
    CPU_OFFLOAD = "cpu_offload"          # Very low memory: keep experts on CPU


class CompressionType(Enum):
    """Expert compression types"""
    NONE = "none"
    INT8_DYNAMIC = "int8_dynamic"
    INT4_GROUPED = "int4_grouped"
    STRUCTURED_SPARSE = "structured_sparse"


class ExpertCache:
    """LRU cache for expert weights with memory management"""
    
    def __init__(self, max_size_mb: float):
        self.max_size_mb = max_size_mb
        self.current_size_mb = 0.0
        self.cache = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.Lock()
    
    def get(self, expert_id: str) -> Optional[torch.Tensor]:
        """Get expert weights from cache"""
        with self.lock:
            if expert_id in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(expert_id)
                self.cache[expert_id] = value
                self.hit_count += 1
                return value
            else:
                self.miss_count += 1
                return None
    
    def put(self, expert_id: str, weights: torch.Tensor) -> bool:
        """Put expert weights in cache, returns True if successfully cached"""
        with self.lock:
            weight_size_mb = weights.numel() * weights.element_size() / (1024**2)
            
            # Check if we can fit this expert
            if weight_size_mb > self.max_size_mb:
                return False
            
            # Evict until we have space
            while self.current_size_mb + weight_size_mb > self.max_size_mb and self.cache:
                oldest_id, oldest_weights = self.cache.popitem(last=False)
                self.current_size_mb -= oldest_weights.numel() * oldest_weights.element_size() / (1024**2)
            
            # Add new expert
            self.cache[expert_id] = weights.clone()
            self.current_size_mb += weight_size_mb
            return True
    
    def clear(self):
        """Clear the cache"""
        with self.lock:
            self.cache.clear()
            self.current_size_mb = 0.0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'hit_rate': hit_rate,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'current_size_mb': self.current_size_mb,
            'max_size_mb': self.max_size_mb,
            'num_experts_cached': len(self.cache)
        }


class ExpertCompressor:
    """Compress expert weights to save memory"""
    
    @staticmethod
    def compress(weights: torch.Tensor, compression_type: CompressionType) -> Tuple[torch.Tensor, Dict]:
        """Compress expert weights"""
        if compression_type == CompressionType.NONE:
            return weights, {}
        
        elif compression_type == CompressionType.INT8_DYNAMIC:
            return ExpertCompressor._compress_int8_dynamic(weights)
        
        elif compression_type == CompressionType.INT4_GROUPED:
            return ExpertCompressor._compress_int4_grouped(weights)
        
        elif compression_type == CompressionType.STRUCTURED_SPARSE:
            return ExpertCompressor._compress_structured_sparse(weights)
        
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
    
    @staticmethod
    def decompress(compressed_weights: torch.Tensor, metadata: Dict, compression_type: CompressionType) -> torch.Tensor:
        """Decompress expert weights"""
        if compression_type == CompressionType.NONE:
            return compressed_weights
        
        elif compression_type == CompressionType.INT8_DYNAMIC:
            return ExpertCompressor._decompress_int8_dynamic(compressed_weights, metadata)
        
        elif compression_type == CompressionType.INT4_GROUPED:
            return ExpertCompressor._decompress_int4_grouped(compressed_weights, metadata)
        
        elif compression_type == CompressionType.STRUCTURED_SPARSE:
            return ExpertCompressor._decompress_structured_sparse(compressed_weights, metadata)
        
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
    
    @staticmethod
    def _compress_int8_dynamic(weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Dynamic INT8 quantization"""
        # Compute scale per tensor
        scale = weights.abs().max() / 127.0
        quantized = torch.round(weights / scale).clamp(-128, 127).to(torch.int8)
        
        metadata = {
            'scale': scale,
            'original_shape': weights.shape,
            'original_dtype': weights.dtype
        }
        
        return quantized, metadata
    
    @staticmethod
    def _decompress_int8_dynamic(quantized: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """Decompress INT8 quantized weights"""
        scale = metadata['scale']
        original_dtype = metadata['original_dtype']
        
        return (quantized.to(torch.float32) * scale).to(original_dtype)
    
    @staticmethod
    def _compress_int4_grouped(weights: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, Dict]:
        """INT4 grouped quantization"""
        original_shape = weights.shape
        weights_flat = weights.flatten()
        
        # Pad to group size
        pad_size = (group_size - len(weights_flat) % group_size) % group_size
        if pad_size > 0:
            weights_flat = torch.cat([weights_flat, torch.zeros(pad_size, device=weights.device)])
        
        # Reshape into groups
        weights_grouped = weights_flat.view(-1, group_size)
        
        # Compute scales per group
        scales = weights_grouped.abs().max(dim=1)[0] / 7.0  # 4-bit range: -8 to 7
        scales = scales.unsqueeze(1)
        
        # Quantize
        quantized = torch.round(weights_grouped / scales).clamp(-8, 7)
        
        # Pack two 4-bit values into one byte
        quantized_even = quantized[:, ::2].to(torch.int8)
        quantized_odd = quantized[:, 1::2].to(torch.int8)
        packed = (quantized_even & 0xF) | ((quantized_odd & 0xF) << 4)
        
        metadata = {
            'scales': scales.squeeze(1),
            'original_shape': original_shape,
            'original_dtype': weights.dtype,
            'group_size': group_size,
            'pad_size': pad_size
        }
        
        return packed, metadata
    
    @staticmethod
    def _decompress_int4_grouped(packed: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """Decompress INT4 grouped quantization"""
        scales = metadata['scales']
        original_shape = metadata['original_shape']
        original_dtype = metadata['original_dtype']
        group_size = metadata['group_size']
        pad_size = metadata['pad_size']
        
        # Unpack 4-bit values
        quantized_even = (packed & 0xF).to(torch.int8)
        quantized_odd = ((packed >> 4) & 0xF).to(torch.int8)
        
        # Handle sign extension for 4-bit values
        quantized_even = torch.where(quantized_even > 7, quantized_even - 16, quantized_even)
        quantized_odd = torch.where(quantized_odd > 7, quantized_odd - 16, quantized_odd)
        
        # Interleave even and odd values
        num_groups, half_group = quantized_even.shape
        quantized = torch.zeros(num_groups, group_size, device=packed.device, dtype=torch.float32)
        quantized[:, ::2] = quantized_even.float()
        quantized[:, 1::2] = quantized_odd.float()
        
        # Dequantize
        scales = scales.unsqueeze(1)
        dequantized = quantized * scales
        
        # Flatten and remove padding
        dequantized_flat = dequantized.flatten()
        if pad_size > 0:
            dequantized_flat = dequantized_flat[:-pad_size]
        
        return dequantized_flat.view(original_shape).to(original_dtype)
    
    @staticmethod
    def _compress_structured_sparse(weights: torch.Tensor, sparsity: float = 0.5) -> Tuple[torch.Tensor, Dict]:
        """Structured sparsity compression (2:4 or magnitude-based)"""
        # Simple magnitude-based pruning for now
        threshold = torch.quantile(weights.abs(), sparsity)
        mask = weights.abs() >= threshold
        
        # Store only non-zero values and their indices
        non_zero_values = weights[mask]
        non_zero_indices = torch.nonzero(mask, as_tuple=False)
        
        metadata = {
            'indices': non_zero_indices,
            'original_shape': weights.shape,
            'original_dtype': weights.dtype,
            'sparsity': sparsity
        }
        
        return non_zero_values, metadata
    
    @staticmethod
    def _decompress_structured_sparse(values: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """Decompress structured sparse weights"""
        indices = metadata['indices']
        original_shape = metadata['original_shape']
        original_dtype = metadata['original_dtype']
        
        # Reconstruct sparse tensor
        sparse_weights = torch.zeros(original_shape, device=values.device, dtype=original_dtype)
        sparse_weights[indices[:, 0], indices[:, 1]] = values
        
        return sparse_weights


class AdaptiveMemoryManager:
    """
    Adaptive memory manager that adjusts strategy based on available memory
    """
    
    def __init__(
        self,
        device_profile,
        model_info: Dict,
        expert_weights: Dict[str, torch.Tensor]
    ):
        self.device_profile = device_profile
        self.model_info = model_info
        self.expert_weights = expert_weights
        
        # Determine optimal strategy
        self.buffer_strategy = self._select_buffer_strategy()
        self.compression_type = CompressionType(device_profile.preferred_compression)
        
        # Initialize caches
        gpu_cache_size_mb = device_profile.memory_capacity_gb * 1024 * 0.3  # 30% for expert cache
        self.gpu_cache = ExpertCache(gpu_cache_size_mb)
        
        if device_profile.has_unified_memory:
            unified_cache_size_mb = device_profile.memory_capacity_gb * 1024 * 0.5
            self.unified_cache = ExpertCache(unified_cache_size_mb)
        else:
            self.unified_cache = None
        
        # Compressed storage
        self.compressed_storage = {}
        self.compression_metadata = {}
        
        # Performance tracking
        self.load_times = []
        self.cache_hit_rates = []
        
        # Pre-compress experts if needed
        self._preprocess_experts()
    
    def _select_buffer_strategy(self) -> BufferStrategy:
        """Select optimal buffering strategy based on device capabilities"""
        available_memory_gb = self.device_profile.memory_capacity_gb * 0.8
        expert_memory_mb = self.model_info.get('memory_per_expert_mb', 0)
        max_experts = int(available_memory_gb * 1024 / expert_memory_mb)
        
        if max_experts >= self.model_info.get('num_experts_per_layer', 8):
            return BufferStrategy.DOUBLE_BUFFER
        elif max_experts >= self.device_profile.max_concurrent_experts:
            return BufferStrategy.SINGLE_BUFFER_ASYNC
        elif max_experts >= 2:
            return BufferStrategy.STREAMING
        else:
            return BufferStrategy.CPU_OFFLOAD
    
    def _preprocess_experts(self):
        """Pre-process experts with compression if needed"""
        print(f"Pre-processing experts with {self.compression_type.value} compression...")
        
        for expert_id, weights in self.expert_weights.items():
            if self.compression_type != CompressionType.NONE:
                compressed_weights, metadata = ExpertCompressor.compress(weights, self.compression_type)
                self.compressed_storage[expert_id] = compressed_weights
                self.compression_metadata[expert_id] = metadata
            else:
                self.compressed_storage[expert_id] = weights
                self.compression_metadata[expert_id] = {}
    
    def load_expert(self, expert_id: str, priority: str = "normal") -> torch.Tensor:
        """Load expert weights with adaptive caching strategy"""
        start_time = time.time()
        
        # Check GPU cache first
        weights = self.gpu_cache.get(expert_id)
        if weights is not None:
            self.load_times.append(time.time() - start_time)
            return weights
        
        # Check unified memory cache (if available)
        if self.unified_cache is not None:
            weights = self.unified_cache.get(expert_id)
            if weights is not None:
                # Copy to GPU cache
                self.gpu_cache.put(expert_id, weights)
                self.load_times.append(time.time() - start_time)
                return weights
        
        # Load from compressed storage
        if expert_id in self.compressed_storage:
            compressed_weights = self.compressed_storage[expert_id]
            metadata = self.compression_metadata[expert_id]
            
            # Decompress if needed
            if self.compression_type != CompressionType.NONE:
                weights = ExpertCompressor.decompress(compressed_weights, metadata, self.compression_type)
            else:
                weights = compressed_weights
            
            # Cache in appropriate tier
            if self.unified_cache is not None:
                self.unified_cache.put(expert_id, weights)
            
            self.gpu_cache.put(expert_id, weights)
            self.load_times.append(time.time() - start_time)
            return weights
        
        raise ValueError(f"Expert {expert_id} not found in storage")
    
    def load_experts_batch(self, expert_ids: List[str], priorities: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Load multiple experts efficiently"""
        if priorities is None:
            priorities = ["normal"] * len(expert_ids)
        
        loaded_experts = {}
        
        if self.buffer_strategy == BufferStrategy.DOUBLE_BUFFER:
            # Load all experts in parallel
            for expert_id, priority in zip(expert_ids, priorities):
                loaded_experts[expert_id] = self.load_expert(expert_id, priority)
        
        elif self.buffer_strategy == BufferStrategy.SINGLE_BUFFER_ASYNC:
            # Load high priority first, then others
            high_priority_ids = [eid for eid, p in zip(expert_ids, priorities) if p == "high"]
            normal_priority_ids = [eid for eid, p in zip(expert_ids, priorities) if p != "high"]
            
            for expert_id in high_priority_ids:
                loaded_experts[expert_id] = self.load_expert(expert_id, "high")
            
            for expert_id in normal_priority_ids:
                loaded_experts[expert_id] = self.load_expert(expert_id, "normal")
        
        elif self.buffer_strategy == BufferStrategy.STREAMING:
            # Load only the most important experts
            sorted_experts = sorted(zip(expert_ids, priorities), key=lambda x: x[1] == "high", reverse=True)
            max_load = min(len(expert_ids), self.device_profile.max_concurrent_experts)
            
            for expert_id, priority in sorted_experts[:max_load]:
                loaded_experts[expert_id] = self.load_expert(expert_id, priority)
        
        elif self.buffer_strategy == BufferStrategy.CPU_OFFLOAD:
            # Keep experts on CPU, only move to GPU when needed
            for expert_id in expert_ids[:2]:  # Load at most 2
                loaded_experts[expert_id] = self.load_expert(expert_id)
        
        return loaded_experts
    
    def prefetch_experts(self, predicted_expert_ids: List[str], confidence: float):
        """Prefetch experts based on speculation"""
        if confidence < 0.5:
            return  # Don't prefetch if confidence is low
        
        # Adjust prefetch aggressiveness based on confidence and device
        max_prefetch = int(self.device_profile.max_concurrent_experts * confidence * self.device_profile.speculation_aggressiveness)
        
        for expert_id in predicted_expert_ids[:max_prefetch]:
            try:
                self.load_expert(expert_id, priority="low")
            except Exception as e:
                print(f"Prefetch failed for expert {expert_id}: {e}")
    
    def get_memory_stats(self) -> Dict:
        """Get comprehensive memory statistics"""
        gpu_stats = self.gpu_cache.get_stats()
        
        stats = {
            'buffer_strategy': self.buffer_strategy.value,
            'compression_type': self.compression_type.value,
            'gpu_cache': gpu_stats,
            'avg_load_time_ms': np.mean(self.load_times) * 1000 if self.load_times else 0.0,
            'total_loads': len(self.load_times)
        }
        
        if self.unified_cache is not None:
            stats['unified_cache'] = self.unified_cache.get_stats()
        
        # Add compression statistics
        if self.compression_type != CompressionType.NONE:
            original_size_mb = sum(w.numel() * 4 / (1024**2) for w in self.expert_weights.values())
            compressed_size_mb = sum(w.numel() * w.element_size() / (1024**2) for w in self.compressed_storage.values())
            stats['compression_ratio'] = original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 1.0
            stats['original_size_mb'] = original_size_mb
            stats['compressed_size_mb'] = compressed_size_mb
        
        return stats
    
    def optimize_memory_usage(self):
        """Dynamically optimize memory usage based on performance"""
        gpu_stats = self.gpu_cache.get_stats()
        
        # If hit rate is low, increase cache size
        if gpu_stats['hit_rate'] < 0.6 and gpu_stats['current_size_mb'] < gpu_stats['max_size_mb'] * 0.8:
            self.gpu_cache.max_size_mb *= 1.2
        
        # If hit rate is very high, we might be able to reduce cache size
        elif gpu_stats['hit_rate'] > 0.9 and gpu_stats['current_size_mb'] > gpu_stats['max_size_mb'] * 0.2:
            self.gpu_cache.max_size_mb *= 0.9
        
        # Adjust prefetch aggressiveness based on load times
        if self.load_times:
            avg_load_time = np.mean(self.load_times[-100:])  # Last 100 loads
            if avg_load_time > 0.01:  # 10ms threshold
                self.device_profile.speculation_aggressiveness = min(1.0, self.device_profile.speculation_aggressiveness * 1.1)
            elif avg_load_time < 0.005:  # 5ms threshold
                self.device_profile.speculation_aggressiveness = max(0.1, self.device_profile.speculation_aggressiveness * 0.9)


def create_memory_manager(device_profile, model, expert_weights: Dict[str, torch.Tensor]) -> AdaptiveMemoryManager:
    """Factory function to create adaptive memory manager"""
    model_info = model.get_model_info()
    
    return AdaptiveMemoryManager(
        device_profile=device_profile,
        model_info=model_info,
        expert_weights=expert_weights
    )


if __name__ == "__main__":
    # Test compression
    print("Testing compression...")
    
    # Create test expert weights
    test_weights = torch.randn(1024, 2048)  # 8MB expert
    
    for compression_type in [CompressionType.INT8_DYNAMIC, CompressionType.INT4_GROUPED]:
        compressed, metadata = ExpertCompressor.compress(test_weights, compression_type)
        decompressed = ExpertCompressor.decompress(compressed, metadata, compression_type)
        
        compression_ratio = test_weights.numel() * 4 / (compressed.numel() * compressed.element_size())
        error = torch.mean(torch.abs(test_weights - decompressed))
        
        print(f"{compression_type.value}:")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Reconstruction error: {error:.6f}")
        print(f"  Original size: {test_weights.numel() * 4 / (1024**2):.1f} MB")
        print(f"  Compressed size: {compressed.numel() * compressed.element_size() / (1024**2):.1f} MB")
        print()
    
    print("Compression test completed!")