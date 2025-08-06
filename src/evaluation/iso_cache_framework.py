#!/usr/bin/env python3

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, OrderedDict
import time

class IsoCacheFramework:
    """
    Iso-cache constraint system ensuring fair comparison across all prefetching strategies.
    All strategies operate under identical cache size and allocation constraints.
    """
    
    def __init__(self, total_cache_size_mb: float, expert_size_mb: float = 2.5):
        self.total_cache_size_mb = total_cache_size_mb
        self.expert_size_mb = expert_size_mb
        
        # Fixed cache hierarchy allocation (matches existing evaluation)
        self.l1_size_mb = total_cache_size_mb * 0.4  # 40% L1 (fastest)
        self.l2_size_mb = total_cache_size_mb * 0.4  # 40% L2 (medium)
        self.l3_size_mb = total_cache_size_mb * 0.2  # 20% L3 (slowest)
        
        # Expert capacity per cache level
        self.l1_capacity = int(self.l1_size_mb / expert_size_mb)
        self.l2_capacity = int(self.l2_size_mb / expert_size_mb)
        self.l3_capacity = int(self.l3_size_mb / expert_size_mb)
        
        # Cache state tracking
        self.l1_cache: OrderedDict = OrderedDict()  # LRU ordering
        self.l2_cache: OrderedDict = OrderedDict()
        self.l3_cache: OrderedDict = OrderedDict()
        
        # Performance tracking
        self.l1_hits = 0
        self.l2_hits = 0
        self.l3_hits = 0
        self.misses = 0
        
        # Cache access latencies (ms)
        self.l1_latency = 0.1
        self.l2_latency = 0.5
        self.l3_latency = 2.0
        self.memory_latency = 10.0  # CPU memory access
        
    def reset_metrics(self):
        """Reset all performance tracking metrics"""
        self.l1_hits = 0
        self.l2_hits = 0
        self.l3_hits = 0
        self.misses = 0
        
    def get_cache_info(self) -> Dict:
        """Return current cache configuration and utilization"""
        return {
            'total_size_mb': self.total_cache_size_mb,
            'l1_size_mb': self.l1_size_mb,
            'l2_size_mb': self.l2_size_mb,
            'l3_size_mb': self.l3_size_mb,
            'l1_capacity': self.l1_capacity,
            'l2_capacity': self.l2_capacity,
            'l3_capacity': self.l3_capacity,
            'l1_utilization': len(self.l1_cache) / self.l1_capacity,
            'l2_utilization': len(self.l2_cache) / self.l2_capacity,
            'l3_utilization': len(self.l3_cache) / self.l3_capacity,
        }
        
    def access_expert(self, expert_id: int) -> Tuple[float, str]:
        """
        Access an expert through the cache hierarchy.
        Returns (latency, cache_level_hit)
        """
        # Check L1 cache first
        if expert_id in self.l1_cache:
            self.l1_hits += 1
            # Move to front (LRU update)
            self.l1_cache.move_to_end(expert_id)
            return self.l1_latency, 'L1'
            
        # Check L2 cache
        if expert_id in self.l2_cache:
            self.l2_hits += 1
            # Promote to L1 cache
            self._promote_to_l1(expert_id)
            self.l2_cache.move_to_end(expert_id)
            return self.l2_latency, 'L2'
            
        # Check L3 cache
        if expert_id in self.l3_cache:
            self.l3_hits += 1
            # Promote to L2 cache
            self._promote_to_l2(expert_id)
            self.l3_cache.move_to_end(expert_id)
            return self.l3_latency, 'L3'
            
        # Cache miss - load from memory
        self.misses += 1
        self._load_to_l3(expert_id)
        return self.memory_latency, 'MEMORY'
        
    def prefetch_expert(self, expert_id: int, target_level: str = 'L3') -> bool:
        """
        Prefetch an expert to specified cache level.
        Returns True if prefetch was successful, False if already cached higher.
        """
        # Check if already in higher-priority cache
        if expert_id in self.l1_cache or expert_id in self.l2_cache:
            return False
            
        if target_level == 'L1':
            if expert_id not in self.l1_cache:
                self._load_to_l1(expert_id)
                return True
        elif target_level == 'L2':
            if expert_id not in self.l2_cache:
                self._load_to_l2(expert_id)
                return True
        else:  # L3
            if expert_id not in self.l3_cache:
                self._load_to_l3(expert_id)
                return True
                
        return False
        
    def _promote_to_l1(self, expert_id: int):
        """Promote expert from L2/L3 to L1 cache"""
        # Remove from lower levels
        self.l2_cache.pop(expert_id, None)
        self.l3_cache.pop(expert_id, None)
        
        # Add to L1 with eviction if needed
        self._load_to_l1(expert_id)
        
    def _promote_to_l2(self, expert_id: int):
        """Promote expert from L3 to L2 cache"""
        self.l3_cache.pop(expert_id, None)
        self._load_to_l2(expert_id)
        
    def _load_to_l1(self, expert_id: int):
        """Load expert to L1 cache with LRU eviction"""
        if len(self.l1_cache) >= self.l1_capacity:
            # Evict LRU item to L2
            evicted_id, _ = self.l1_cache.popitem(last=False)
            self._load_to_l2(evicted_id)
            
        self.l1_cache[expert_id] = time.time()
        
    def _load_to_l2(self, expert_id: int):
        """Load expert to L2 cache with LRU eviction"""
        if len(self.l2_cache) >= self.l2_capacity:
            # Evict LRU item to L3
            evicted_id, _ = self.l2_cache.popitem(last=False)
            self._load_to_l3(evicted_id)
            
        self.l2_cache[expert_id] = time.time()
        
    def _load_to_l3(self, expert_id: int):
        """Load expert to L3 cache with LRU eviction"""
        if len(self.l3_cache) >= self.l3_capacity:
            # Evict LRU item (completely removed from cache)
            self.l3_cache.popitem(last=False)
            
        self.l3_cache[expert_id] = time.time()
        
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive cache performance metrics"""
        total_accesses = self.l1_hits + self.l2_hits + self.l3_hits + self.misses
        
        if total_accesses == 0:
            return {
                'total_accesses': 0,
                'l1_hit_rate': 0.0,
                'l2_hit_rate': 0.0,
                'l3_hit_rate': 0.0,
                'overall_hit_rate': 0.0,
                'miss_rate': 0.0,
                'average_latency': 0.0
            }
            
        l1_hit_rate = self.l1_hits / total_accesses
        l2_hit_rate = self.l2_hits / total_accesses
        l3_hit_rate = self.l3_hits / total_accesses
        miss_rate = self.misses / total_accesses
        
        # Calculate weighted average latency
        avg_latency = (
            self.l1_hits * self.l1_latency +
            self.l2_hits * self.l2_latency +
            self.l3_hits * self.l3_latency +
            self.misses * self.memory_latency
        ) / total_accesses
        
        return {
            'total_accesses': total_accesses,
            'l1_hits': self.l1_hits,
            'l2_hits': self.l2_hits,
            'l3_hits': self.l3_hits,
            'misses': self.misses,
            'l1_hit_rate': l1_hit_rate,
            'l2_hit_rate': l2_hit_rate,
            'l3_hit_rate': l3_hit_rate,
            'overall_hit_rate': 1.0 - miss_rate,
            'miss_rate': miss_rate,
            'average_latency': avg_latency
        }
        
    def clear_cache(self):
        """Clear all cache levels"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.l3_cache.clear()
        
    def get_cached_experts(self) -> Dict[str, Set[int]]:
        """Return set of experts in each cache level"""
        return {
            'L1': set(self.l1_cache.keys()),
            'L2': set(self.l2_cache.keys()),
            'L3': set(self.l3_cache.keys())
        }

class BatchSizeAwareCacheFramework(IsoCacheFramework):
    """
    Extended iso-cache framework that accounts for batch size effects on caching.
    Larger batches may benefit from different caching strategies.
    """
    
    def __init__(self, total_cache_size_mb: float, expert_size_mb: float = 2.5):
        super().__init__(total_cache_size_mb, expert_size_mb)
        self.batch_size = 1
        self.batch_access_patterns = defaultdict(list)
        
    def set_batch_size(self, batch_size: int):
        """Configure framework for specific batch size"""
        self.batch_size = batch_size
        
    def batch_access_experts(self, expert_ids: List[int]) -> Tuple[float, Dict]:
        """
        Access multiple experts simultaneously (batch processing).
        Returns total latency and detailed access information.
        """
        total_latency = 0.0
        access_details = {
            'L1': [],
            'L2': [],
            'L3': [],
            'MEMORY': []
        }
        
        # Process all expert accesses
        for expert_id in expert_ids:
            latency, level = self.access_expert(expert_id)
            total_latency += latency
            access_details[level].append(expert_id)
            
        # Store batch access pattern for analysis
        self.batch_access_patterns[self.batch_size].extend(expert_ids)
        
        return total_latency, access_details
        
    def get_batch_statistics(self) -> Dict:
        """Analyze access patterns across different batch sizes"""
        stats = {}
        for batch_size, accesses in self.batch_access_patterns.items():
            if accesses:
                unique_experts = len(set(accesses))
                total_accesses = len(accesses)
                reuse_ratio = 1.0 - (unique_experts / total_accesses)
                
                stats[batch_size] = {
                    'unique_experts': unique_experts,
                    'total_accesses': total_accesses,
                    'reuse_ratio': reuse_ratio,
                    'avg_experts_per_batch': total_accesses / (total_accesses / batch_size) if total_accesses > 0 else 0
                }
                
        return stats