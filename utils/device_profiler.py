"""
Device profiler for RTX 3090 and hardware adaptation
"""

import torch
import psutil
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import subprocess
import platform


@dataclass
class DeviceProfile:
    """Device capability profile"""
    device_name: str
    memory_capacity_gb: float
    memory_bandwidth_gbps: float
    compute_capability: str
    has_unified_memory: bool
    max_concurrent_experts: int
    speculation_aggressiveness: float
    preferred_compression: str
    optimal_batch_size: int
    memory_efficiency_score: float


class DeviceProfiler:
    """Profile device capabilities and create optimization recommendations"""
    
    def __init__(self):
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.device_props = None
        if self.device is not None:
            self.device_props = torch.cuda.get_device_properties(self.device)
    
    def profile_device(self) -> DeviceProfile:
        """Comprehensive device profiling"""
        if not torch.cuda.is_available():
            return self._get_cpu_profile()
        
        device_name = self.device_props.name
        memory_capacity = self.device_props.total_memory / (1024**3)  # GB
        
        # Get more detailed device info
        memory_bandwidth = self._estimate_memory_bandwidth()
        compute_capability = f"{self.device_props.major}.{self.device_props.minor}"
        
        # Device-specific optimizations
        if "RTX 3090" in device_name:
            return self._get_rtx3090_profile(memory_capacity, memory_bandwidth, compute_capability)
        elif "Jetson" in device_name or "Orin" in device_name:
            return self._get_jetson_profile(memory_capacity, memory_bandwidth, compute_capability)
        elif "A100" in device_name:
            return self._get_a100_profile(memory_capacity, memory_bandwidth, compute_capability)
        else:
            return self._get_generic_profile(device_name, memory_capacity, memory_bandwidth, compute_capability)
    
    def _estimate_memory_bandwidth(self) -> float:
        """Estimate memory bandwidth through benchmarking"""
        if not torch.cuda.is_available():
            return 50.0  # Default for CPU
        
        # Quick memory bandwidth test
        device = torch.cuda.current_device()
        size = 256 * 1024 * 1024  # 256MB
        
        # Create test tensors
        a = torch.randn(size // 4, device=device, dtype=torch.float32)
        b = torch.randn(size // 4, device=device, dtype=torch.float32)
        
        # Warmup
        for _ in range(5):
            c = a + b
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            c = a + b
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate bandwidth (GB/s)
        total_bytes = size * 3 * 100  # a, b, c for 100 iterations
        bandwidth = total_bytes / (end_time - start_time) / (1024**3)
        
        return bandwidth
    
    def _get_rtx3090_profile(self, memory_gb: float, bandwidth: float, compute_cap: str) -> DeviceProfile:
        """RTX 3090 specific optimizations"""
        return DeviceProfile(
            device_name="RTX 3090",
            memory_capacity_gb=memory_gb,
            memory_bandwidth_gbps=bandwidth,
            compute_capability=compute_cap,
            has_unified_memory=False,
            max_concurrent_experts=6,  # Can fit ~6 experts comfortably
            speculation_aggressiveness=0.7,
            preferred_compression="int8_dynamic",
            optimal_batch_size=8,
            memory_efficiency_score=0.8
        )
    
    def _get_jetson_profile(self, memory_gb: float, bandwidth: float, compute_cap: str) -> DeviceProfile:
        """Jetson device optimizations"""
        return DeviceProfile(
            device_name="Jetson AGX Orin",
            memory_capacity_gb=memory_gb,
            memory_bandwidth_gbps=bandwidth,
            compute_capability=compute_cap,
            has_unified_memory=True,
            max_concurrent_experts=4,
            speculation_aggressiveness=0.9,  # Higher due to unified memory
            preferred_compression="int4_grouped",
            optimal_batch_size=4,
            memory_efficiency_score=0.9
        )
    
    def _get_a100_profile(self, memory_gb: float, bandwidth: float, compute_cap: str) -> DeviceProfile:
        """A100 optimizations (reference)"""
        return DeviceProfile(
            device_name="A100",
            memory_capacity_gb=memory_gb,
            memory_bandwidth_gbps=bandwidth,
            compute_capability=compute_cap,
            has_unified_memory=False,
            max_concurrent_experts=16,
            speculation_aggressiveness=0.5,  # Less aggressive due to large memory
            preferred_compression="none",
            optimal_batch_size=32,
            memory_efficiency_score=0.95
        )
    
    def _get_generic_profile(self, name: str, memory_gb: float, bandwidth: float, compute_cap: str) -> DeviceProfile:
        """Generic GPU profile"""
        # Estimate capabilities based on memory
        if memory_gb > 20:
            max_experts = 8
            batch_size = 16
            compression = "int8_dynamic"
        elif memory_gb > 10:
            max_experts = 4
            batch_size = 8
            compression = "int8_dynamic"
        else:
            max_experts = 2
            batch_size = 4
            compression = "int4_grouped"
        
        return DeviceProfile(
            device_name=name,
            memory_capacity_gb=memory_gb,
            memory_bandwidth_gbps=bandwidth,
            compute_capability=compute_cap,
            has_unified_memory=False,
            max_concurrent_experts=max_experts,
            speculation_aggressiveness=0.6,
            preferred_compression=compression,
            optimal_batch_size=batch_size,
            memory_efficiency_score=0.7
        )
    
    def _get_cpu_profile(self) -> DeviceProfile:
        """CPU fallback profile"""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        return DeviceProfile(
            device_name="CPU",
            memory_capacity_gb=memory_gb,
            memory_bandwidth_gbps=50.0,
            compute_capability="CPU",
            has_unified_memory=True,
            max_concurrent_experts=2,
            speculation_aggressiveness=0.8,
            preferred_compression="int8_dynamic",
            optimal_batch_size=2,
            memory_efficiency_score=0.6
        )
    
    def benchmark_model_memory(self, model, batch_size: int = 1, seq_len: int = 128) -> Dict:
        """Benchmark model memory usage"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        device = torch.cuda.current_device()
        torch.cuda.empty_cache()
        
        # Measure baseline memory
        baseline_memory = torch.cuda.memory_allocated(device)
        
        # Move model to device
        model = model.cuda()
        model_memory = torch.cuda.memory_allocated(device) - baseline_memory
        
        # Create sample input
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        
        # Measure forward pass memory
        with torch.no_grad():
            outputs = model(input_ids)
        
        forward_memory = torch.cuda.memory_allocated(device) - baseline_memory
        peak_memory = torch.cuda.max_memory_allocated(device) - baseline_memory
        
        # Clean up
        del outputs, input_ids
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        return {
            "model_memory_mb": model_memory / (1024**2),
            "forward_memory_mb": forward_memory / (1024**2),
            "peak_memory_mb": peak_memory / (1024**2),
            "available_memory_mb": (torch.cuda.get_device_properties(device).total_memory - peak_memory) / (1024**2)
        }
    
    def recommend_configuration(self, model_info: Dict, device_profile: DeviceProfile) -> Dict:
        """Recommend optimal configuration based on device and model"""
        model_memory_mb = model_info.get('total_parameters', 0) * 4 / (1024**2)  # FP32
        expert_memory_mb = model_info.get('memory_per_expert_mb', 0)
        
        # Calculate how many experts can fit in memory
        available_memory_gb = device_profile.memory_capacity_gb * 0.8  # Leave 20% buffer
        available_memory_mb = available_memory_gb * 1024
        
        base_model_memory = model_memory_mb - model_info.get('expert_parameters', 0) * 4 / (1024**2)
        remaining_memory = available_memory_mb - base_model_memory
        
        max_experts_in_memory = int(remaining_memory / expert_memory_mb)
        
        # Adjust based on device capabilities
        recommended_experts = min(max_experts_in_memory, device_profile.max_concurrent_experts)
        
        # Compression recommendation
        if recommended_experts < model_info.get('num_experts_per_layer', 8):
            compression_needed = True
            if device_profile.preferred_compression == "int4_grouped":
                compression_ratio = 4
            elif device_profile.preferred_compression == "int8_dynamic":
                compression_ratio = 2
            else:
                compression_ratio = 1
        else:
            compression_needed = False
            compression_ratio = 1
        
        return {
            "max_experts_in_memory": recommended_experts,
            "compression_needed": compression_needed,
            "compression_ratio": compression_ratio,
            "compression_type": device_profile.preferred_compression,
            "optimal_batch_size": device_profile.optimal_batch_size,
            "speculation_aggressiveness": device_profile.speculation_aggressiveness,
            "memory_utilization": (base_model_memory + recommended_experts * expert_memory_mb / compression_ratio) / available_memory_mb,
            "caching_strategy": "hierarchical" if device_profile.has_unified_memory else "gpu_only"
        }
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        }
        
        # CPU info
        info["cpu"] = {
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "frequency": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown"
        }
        
        # Memory info
        memory = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_percent": memory.percent
        }
        
        # GPU info
        if torch.cuda.is_available():
            info["gpu"] = {
                "name": self.device_props.name,
                "compute_capability": f"{self.device_props.major}.{self.device_props.minor}",
                "total_memory_gb": self.device_props.total_memory / (1024**3),
                "multiprocessor_count": self.device_props.multi_processor_count,
                "cuda_version": torch.version.cuda
            }
        else:
            info["gpu"] = {"available": False}
        
        return info


def profile_current_device() -> DeviceProfile:
    """Quick function to profile current device"""
    profiler = DeviceProfiler()
    return profiler.profile_device()


if __name__ == "__main__":
    # Profile current device
    profiler = DeviceProfiler()
    
    print("=== System Information ===")
    system_info = profiler.get_system_info()
    print(json.dumps(system_info, indent=2))
    
    print("\n=== Device Profile ===")
    device_profile = profiler.profile_device()
    print(f"Device: {device_profile.device_name}")
    print(f"Memory: {device_profile.memory_capacity_gb:.1f} GB")
    print(f"Bandwidth: {device_profile.memory_bandwidth_gbps:.1f} GB/s")
    print(f"Max concurrent experts: {device_profile.max_concurrent_experts}")
    print(f"Speculation aggressiveness: {device_profile.speculation_aggressiveness}")
    print(f"Preferred compression: {device_profile.preferred_compression}")
    print(f"Optimal batch size: {device_profile.optimal_batch_size}")
    
    # Test with a small model
    try:
        import sys
        sys.path.append('../models')
        from small_switch_transformer import create_small_switch_model
        
        print("\n=== Model Memory Benchmark ===")
        model = create_small_switch_model()
        model_info = model.get_model_info()
        
        memory_usage = profiler.benchmark_model_memory(model, batch_size=2, seq_len=128)
        print(f"Model memory: {memory_usage['model_memory_mb']:.1f} MB")
        print(f"Forward pass memory: {memory_usage['forward_memory_mb']:.1f} MB")
        print(f"Peak memory: {memory_usage['peak_memory_mb']:.1f} MB")
        
        print("\n=== Configuration Recommendations ===")
        recommendations = profiler.recommend_configuration(model_info, device_profile)
        print(json.dumps(recommendations, indent=2))
        
    except ImportError:
        print("Could not import model for testing")