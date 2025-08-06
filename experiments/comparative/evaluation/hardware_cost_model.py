#!/usr/bin/env python3

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class DeviceType(Enum):
    """Hardware device types with different characteristics"""
    RTX_4090 = "rtx_4090"
    A100_40GB = "a100_40gb"
    H100_80GB = "h100_80gb"
    JETSON_ORIN = "jetson_orin"
    CPU_ONLY = "cpu_only"

@dataclass
class HardwareSpec:
    """Hardware specification for cost modeling"""
    device_type: DeviceType
    gpu_memory_gb: float
    cpu_memory_gb: float
    pcie_bandwidth_gbps: float  # CPU-GPU transfer bandwidth
    gpu_memory_bandwidth_gbps: float  # HBM bandwidth
    cpu_memory_bandwidth_gbps: float  # DDR bandwidth
    compute_units: int  # CUDA cores, Tensor cores, etc.
    memory_latency_ms: float  # Base memory access latency
    
    # Real-world measured parameters
    expert_load_latency_ms: float  # Measured expert loading time
    memory_contention_factor: float  # Memory bandwidth contention multiplier

# Predefined hardware configurations based on research literature
HARDWARE_CONFIGS = {
    DeviceType.RTX_4090: HardwareSpec(
        device_type=DeviceType.RTX_4090,
        gpu_memory_gb=24,
        cpu_memory_gb=64,
        pcie_bandwidth_gbps=64,  # PCIe 4.0 x16
        gpu_memory_bandwidth_gbps=1008,  # GDDR6X
        cpu_memory_bandwidth_gbps=51.2,  # DDR4-3200
        compute_units=16384,
        memory_latency_ms=0.1,
        expert_load_latency_ms=2.5,  # From ExpertFlow paper
        memory_contention_factor=1.3
    ),
    
    DeviceType.A100_40GB: HardwareSpec(
        device_type=DeviceType.A100_40GB,
        gpu_memory_gb=40,
        cpu_memory_gb=128,
        pcie_bandwidth_gbps=64,
        gpu_memory_bandwidth_gbps=1555,  # HBM2e
        cpu_memory_bandwidth_gbps=68.3,  # DDR4-4266
        compute_units=6912,
        memory_latency_ms=0.08,
        expert_load_latency_ms=1.8,
        memory_contention_factor=1.2
    ),
    
    DeviceType.H100_80GB: HardwareSpec(
        device_type=DeviceType.H100_80GB,
        gpu_memory_gb=80,
        cpu_memory_gb=256,
        pcie_bandwidth_gbps=64,
        gpu_memory_bandwidth_gbps=2039,  # HBM3
        cpu_memory_bandwidth_gbps=85.4,  # DDR5-5333
        compute_units=16896,
        memory_latency_ms=0.06,
        expert_load_latency_ms=1.2,
        memory_contention_factor=1.1
    ),
    
    DeviceType.JETSON_ORIN: HardwareSpec(
        device_type=DeviceType.JETSON_ORIN,
        gpu_memory_gb=32,  # Shared memory
        cpu_memory_gb=32,
        pcie_bandwidth_gbps=16,  # Limited by integrated design
        gpu_memory_bandwidth_gbps=204,  # LPDDR5
        cpu_memory_bandwidth_gbps=204,
        compute_units=2048,
        memory_latency_ms=0.2,
        expert_load_latency_ms=8.5,  # From HOBBIT paper
        memory_contention_factor=1.8
    ),
    
    DeviceType.CPU_ONLY: HardwareSpec(
        device_type=DeviceType.CPU_ONLY,
        gpu_memory_gb=0,
        cpu_memory_gb=128,
        pcie_bandwidth_gbps=0,
        gpu_memory_bandwidth_gbps=0,
        cpu_memory_bandwidth_gbps=68.3,
        compute_units=32,  # CPU cores
        memory_latency_ms=0.5,
        expert_load_latency_ms=15.0,
        memory_contention_factor=2.0
    )
}

class HardwareAwareCostModel:
    """
    Hardware-aware cost modeling for MoE expert prefetching strategies.
    
    Models realistic CPU↔GPU transfer costs, memory bandwidth contention,
    and device-specific performance characteristics based on research literature.
    """
    
    def __init__(self, device_type: DeviceType = DeviceType.RTX_4090, expert_size_mb: float = 2.5):
        self.hardware_spec = HARDWARE_CONFIGS[device_type]
        self.expert_size_mb = expert_size_mb
        
        # Derived parameters
        self.expert_size_bytes = expert_size_mb * 1024 * 1024
        self.experts_per_gb = 1024 / expert_size_mb
        
        # Performance tracking
        self.total_transfer_time = 0.0
        self.total_transfers = 0
        self.memory_contentions = 0
        
        # Dynamic state
        self.current_gpu_utilization = 0.0
        self.current_memory_pressure = 0.0
        
    def calculate_cpu_gpu_transfer_cost(self, num_experts: int, concurrent_operations: int = 1) -> float:
        """
        Calculate realistic CPU→GPU expert transfer latency.
        
        Args:
            num_experts: Number of experts to transfer
            concurrent_operations: Number of concurrent memory operations
            
        Returns:
            Transfer latency in milliseconds
        """
        if num_experts == 0:
            return 0.0
            
        # Base transfer time calculation
        total_bytes = num_experts * self.expert_size_bytes
        base_transfer_time = (total_bytes / (self.hardware_spec.pcie_bandwidth_gbps * 1e9)) * 1000  # ms
        
        # Memory contention modeling
        contention_factor = self.hardware_spec.memory_contention_factor
        if concurrent_operations > 1:
            contention_factor *= (1 + 0.2 * (concurrent_operations - 1))
        
        # Memory bandwidth saturation
        pcie_utilization = min(1.0, total_bytes / (self.hardware_spec.pcie_bandwidth_gbps * 1e9))
        saturation_penalty = 1.0 + (pcie_utilization ** 2) * 0.5
        
        # Device-specific expert loading overhead
        device_overhead = self.hardware_spec.expert_load_latency_ms * num_experts
        
        total_latency = (base_transfer_time * contention_factor * saturation_penalty) + device_overhead
        
        # Update tracking
        self.total_transfer_time += total_latency
        self.total_transfers += num_experts
        if contention_factor > self.hardware_spec.memory_contention_factor:
            self.memory_contentions += 1
        
        return total_latency
    
    def calculate_gpu_memory_access_cost(self, cache_level: str, num_experts: int) -> float:
        """
        Calculate GPU memory access latency for different cache levels.
        
        Args:
            cache_level: 'L1', 'L2', 'L3', or 'MEMORY'
            num_experts: Number of experts accessed
            
        Returns:
            Access latency in milliseconds
        """
        if num_experts == 0:
            return 0.0
            
        # Base latency by cache level (hardware-specific)
        base_latencies = {
            'L1': self.hardware_spec.memory_latency_ms * 0.1,  # GPU L1 cache
            'L2': self.hardware_spec.memory_latency_ms * 0.5,  # GPU L2 cache
            'L3': self.hardware_spec.memory_latency_ms * 2.0,  # GPU memory
            'MEMORY': self.hardware_spec.memory_latency_ms * 10.0  # CPU memory (requires transfer)
        }
        
        base_latency = base_latencies.get(cache_level, base_latencies['MEMORY'])
        
        # Memory bandwidth contention
        total_bytes = num_experts * self.expert_size_bytes
        
        if cache_level == 'MEMORY':
            # CPU memory access requires PCIe transfer
            bandwidth = self.hardware_spec.pcie_bandwidth_gbps * 1e9
        else:
            # GPU memory access
            bandwidth = self.hardware_spec.gpu_memory_bandwidth_gbps * 1e9
            
        bandwidth_latency = (total_bytes / bandwidth) * 1000  # ms
        
        # Memory pressure effects
        pressure_penalty = 1.0 + self.current_memory_pressure * 0.3
        
        return (base_latency + bandwidth_latency) * pressure_penalty
    
    def model_batch_processing_efficiency(self, batch_size: int, experts_per_batch: int) -> Dict[str, float]:
        """
        Model batch processing efficiency effects on hardware utilization.
        
        Args:
            batch_size: Batch size being processed
            experts_per_batch: Average experts used per batch item
            
        Returns:
            Dictionary with efficiency metrics
        """
        # Compute utilization based on batch size
        optimal_batch_size = self.hardware_spec.compute_units // 256  # Rough heuristic
        
        if batch_size <= optimal_batch_size:
            compute_efficiency = batch_size / optimal_batch_size
        else:
            # Diminishing returns for larger batches
            compute_efficiency = 1.0 - (batch_size - optimal_batch_size) * 0.1
            compute_efficiency = max(0.3, compute_efficiency)
        
        # Memory efficiency based on expert reuse
        expert_reuse_ratio = min(1.0, batch_size * experts_per_batch / (batch_size * 8))  # Assuming top-8
        memory_efficiency = 0.5 + 0.5 * expert_reuse_ratio
        
        # Overall hardware efficiency
        overall_efficiency = (compute_efficiency * 0.6) + (memory_efficiency * 0.4)
        
        return {
            'compute_efficiency': compute_efficiency,
            'memory_efficiency': memory_efficiency,
            'overall_efficiency': overall_efficiency,
            'optimal_batch_size': optimal_batch_size
        }
    
    def calculate_prefetch_overlap_benefit(self, computation_time: float, prefetch_time: float, 
                                        overlap_ratio: float = 0.8) -> float:
        """
        Calculate benefit from overlapping computation with prefetching.
        
        Args:
            computation_time: Time for current layer computation (ms)
            prefetch_time: Time required for prefetching (ms)
            overlap_ratio: Fraction of prefetch that can be overlapped
            
        Returns:
            Time saved due to overlap (ms)
        """
        overlappable_time = min(computation_time, prefetch_time * overlap_ratio)
        
        # Hardware-specific overlap efficiency
        if self.hardware_spec.device_type == DeviceType.H100_80GB:
            overlap_efficiency = 0.9  # High-end hardware has better overlap
        elif self.hardware_spec.device_type == DeviceType.A100_40GB:
            overlap_efficiency = 0.85
        elif self.hardware_spec.device_type == DeviceType.RTX_4090:
            overlap_efficiency = 0.8
        else:
            overlap_efficiency = 0.7  # Lower-end hardware
        
        return overlappable_time * overlap_efficiency
    
    def update_dynamic_state(self, gpu_utilization: float, memory_pressure: float):
        """Update dynamic hardware state for more accurate modeling"""
        self.current_gpu_utilization = min(1.0, max(0.0, gpu_utilization))
        self.current_memory_pressure = min(1.0, max(0.0, memory_pressure))
    
    def get_performance_characteristics(self) -> Dict[str, Any]:
        """Get comprehensive hardware performance characteristics"""
        return {
            'device_type': self.hardware_spec.device_type.value,
            'gpu_memory_gb': self.hardware_spec.gpu_memory_gb,
            'pcie_bandwidth_gbps': self.hardware_spec.pcie_bandwidth_gbps,
            'gpu_memory_bandwidth_gbps': self.hardware_spec.gpu_memory_bandwidth_gbps,
            'expert_capacity_gb': self.hardware_spec.gpu_memory_gb * self.experts_per_gb,
            'base_expert_load_ms': self.hardware_spec.expert_load_latency_ms,
            'memory_contention_factor': self.hardware_spec.memory_contention_factor,
            
            # Derived performance metrics
            'experts_per_transfer_ms': 1.0 / self.hardware_spec.expert_load_latency_ms,
            'max_concurrent_experts': int(self.hardware_spec.gpu_memory_gb * self.experts_per_gb * 0.8),
            'recommended_cache_size_mb': self.hardware_spec.gpu_memory_gb * 1024 * 0.6,  # 60% of GPU memory
            
            # Runtime statistics
            'total_transfer_time_ms': self.total_transfer_time,
            'total_experts_transferred': self.total_transfers,
            'average_transfer_time_ms': (
                self.total_transfer_time / self.total_transfers 
                if self.total_transfers > 0 else 0.0
            ),
            'memory_contentions': self.memory_contentions
        }
    
    def estimate_strategy_suitability(self, strategy_characteristics: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate how well a strategy fits the hardware characteristics.
        
        Args:
            strategy_characteristics: Dictionary with strategy properties
            
        Returns:
            Suitability scores (0-1) for different aspects
        """
        prefetch_intensity = strategy_characteristics.get('prefetch_intensity', 0.5)
        memory_requirements = strategy_characteristics.get('memory_requirements_mb', 100)
        prediction_complexity = strategy_characteristics.get('prediction_complexity', 0.5)
        
        # Memory suitability
        memory_ratio = memory_requirements / (self.hardware_spec.gpu_memory_gb * 1024)
        memory_suitability = max(0.0, 1.0 - memory_ratio)
        
        # Bandwidth suitability
        bandwidth_demand = prefetch_intensity * self.expert_size_mb * 10  # Rough estimate
        bandwidth_ratio = bandwidth_demand / self.hardware_spec.pcie_bandwidth_gbps
        bandwidth_suitability = max(0.0, 1.0 - bandwidth_ratio)
        
        # Computational suitability
        compute_demand = prediction_complexity * 100  # Rough metric
        compute_ratio = compute_demand / self.hardware_spec.compute_units
        compute_suitability = max(0.0, 1.0 - compute_ratio)
        
        # Overall suitability
        overall_suitability = (
            memory_suitability * 0.4 +
            bandwidth_suitability * 0.4 +
            compute_suitability * 0.2
        )
        
        return {
            'memory_suitability': memory_suitability,
            'bandwidth_suitability': bandwidth_suitability,
            'compute_suitability': compute_suitability,
            'overall_suitability': overall_suitability
        }
    
    def reset_tracking(self):
        """Reset performance tracking counters"""
        self.total_transfer_time = 0.0
        self.total_transfers = 0
        self.memory_contentions = 0
        self.current_gpu_utilization = 0.0
        self.current_memory_pressure = 0.0

class MultiDeviceCostModel:
    """
    Multi-device cost modeling for comparing strategy performance across hardware.
    """
    
    def __init__(self, device_types: List[DeviceType], expert_size_mb: float = 2.5):
        self.cost_models = {
            device_type: HardwareAwareCostModel(device_type, expert_size_mb)
            for device_type in device_types
        }
        
    def evaluate_strategy_across_devices(self, strategy_characteristics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Evaluate strategy suitability across all configured devices"""
        results = {}
        
        for device_type, cost_model in self.cost_models.items():
            suitability = cost_model.estimate_strategy_suitability(strategy_characteristics)
            performance = cost_model.get_performance_characteristics()
            
            results[device_type.value] = {
                **suitability,
                'device_performance': performance
            }
            
        return results
    
    def recommend_optimal_device(self, strategy_characteristics: Dict[str, Any]) -> Tuple[DeviceType, float]:
        """Recommend optimal device for a given strategy"""
        evaluations = self.evaluate_strategy_across_devices(strategy_characteristics)
        
        best_device = None
        best_score = -1.0
        
        for device_name, metrics in evaluations.items():
            score = metrics['overall_suitability']
            if score > best_score:
                best_score = score
                best_device = DeviceType(device_name)
                
        return best_device, best_score

if __name__ == "__main__":
    # Example usage
    cost_model = HardwareAwareCostModel(DeviceType.RTX_4090)
    
    # Test transfer cost calculation
    transfer_cost = cost_model.calculate_cpu_gpu_transfer_cost(num_experts=32, concurrent_operations=2)
    print(f"Transfer cost for 32 experts: {transfer_cost:.2f}ms")
    
    # Test batch processing efficiency
    batch_efficiency = cost_model.model_batch_processing_efficiency(batch_size=16, experts_per_batch=8)
    print(f"Batch efficiency: {batch_efficiency}")
    
    # Test hardware characteristics
    characteristics = cost_model.get_performance_characteristics()
    print(f"Hardware characteristics: {characteristics}")
    
    # Test multi-device comparison
    multi_device = MultiDeviceCostModel([DeviceType.RTX_4090, DeviceType.A100_40GB, DeviceType.H100_80GB])
    
    strategy_char = {
        'prefetch_intensity': 0.8,
        'memory_requirements_mb': 150, 
        'prediction_complexity': 0.6
    }
    
    device_comparison = multi_device.evaluate_strategy_across_devices(strategy_char)
    optimal_device, score = multi_device.recommend_optimal_device(strategy_char)
    
    print(f"Optimal device: {optimal_device.value} (score: {score:.3f})")