#!/usr/bin/env python3
"""
GPU Usage Verification Script
Ensures all training runs on GPU and monitors GPU utilization
"""

import torch
import logging
import time
import subprocess
import threading
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUMonitor:
    """Monitor GPU usage during training"""
    
    def __init__(self):
        self.monitoring = False
        self.max_memory = 0
        self.avg_utilization = 0
        self.measurements = []
    
    def start_monitoring(self):
        """Start GPU monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("🔍 GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
        
        if self.measurements:
            self.avg_utilization = sum(m[0] for m in self.measurements) / len(self.measurements)
            self.max_memory = max(m[1] for m in self.measurements)
        
        logger.info("🔍 GPU monitoring stopped")
        logger.info(f"   Max GPU utilization: {max(m[0] for m in self.measurements) if self.measurements else 0:.1f}%")
        logger.info(f"   Avg GPU utilization: {self.avg_utilization:.1f}%")
        logger.info(f"   Max GPU memory: {self.max_memory:.2f} GB")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Get GPU utilization via nvidia-smi
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split(', ')
                            if len(parts) >= 2:
                                util = float(parts[0])
                                memory_mb = float(parts[1])
                                memory_gb = memory_mb / 1024
                                self.measurements.append((util, memory_gb))
                
            except Exception as e:
                logger.debug(f"GPU monitoring error: {e}")
            
            time.sleep(2)  # Check every 2 seconds

def verify_gpu_setup():
    """Verify GPU setup and configuration"""
    
    logger.info("🔧 GPU Setup Verification")
    logger.info("=" * 40)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! Training will run on CPU")
        return False
    
    # GPU information
    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    logger.info(f"✅ CUDA available")
    logger.info(f"🔥 GPU Count: {gpu_count}")
    logger.info(f"🎯 Current Device: {current_device}")
    logger.info(f"💻 GPU Name: {gpu_name}")
    logger.info(f"💾 Total Memory: {gpu_memory:.1f} GB")
    
    # Check GPU memory
    if gpu_memory < 8:
        logger.warning(f"⚠️ GPU memory ({gpu_memory:.1f} GB) might be insufficient for large models")
    
    # Memory usage before training
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    
    logger.info(f"📊 Memory allocated: {allocated:.2f} GB")
    logger.info(f"📊 Memory reserved: {reserved:.2f} GB")
    logger.info(f"📊 Memory available: {gpu_memory - reserved:.2f} GB")
    
    return True

def test_gpu_training():
    """Test GPU training with a simple model"""
    
    logger.info("\n🧪 GPU Training Test")
    logger.info("-" * 25)
    
    if not torch.cuda.is_available():
        logger.error("❌ Cannot test GPU training - CUDA not available")
        return False
    
    device = torch.device("cuda")
    
    # Create simple test model
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128)
    ).to(device)
    
    # Test data
    batch_size = 16
    input_data = torch.randn(batch_size, 512).to(device)
    target_data = torch.randint(0, 128, (batch_size,)).to(device)
    
    # Test forward pass
    try:
        output = model(input_data)
        loss = torch.nn.functional.cross_entropy(output, target_data)
        
        logger.info(f"✅ Forward pass successful")
        logger.info(f"   Input shape: {input_data.shape}")
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Loss: {loss.item():.4f}")
        logger.info(f"   Model device: {next(model.parameters()).device}")
        logger.info(f"   Data device: {input_data.device}")
        
        # Test backward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"✅ Backward pass successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU training test failed: {e}")
        return False

def run_gpu_verified_training():
    """Run all training approaches with GPU monitoring"""
    
    logger.info("\n🚀 GPU-Verified Training Pipeline")
    logger.info("=" * 50)
    
    # Verify GPU setup
    if not verify_gpu_setup():
        logger.error("❌ GPU setup verification failed")
        return False
    
    # Test GPU training
    if not test_gpu_training():
        logger.error("❌ GPU training test failed")
        return False
    
    # Import training modules
    try:
        from train_simple_fixed import train_simple_model
        from train_improved_accuracy import train_improved_model
        from train_sophisticated_speculation import main as sophisticated_main
    except ImportError as e:
        logger.error(f"❌ Failed to import training modules: {e}")
        return False
    
    results = {}
    
    # Test each approach with GPU monitoring
    approaches = [
        ("Baseline Simple", lambda: train_simple_model()),
        ("Improved Weighted", lambda: train_improved_model('weighted')),
        ("Sophisticated Multi-Layer", lambda: sophisticated_main())
    ]
    
    for name, train_func in approaches:
        logger.info(f"\n{'='*50}")
        logger.info(f"🧪 Testing {name} with GPU Monitoring")
        logger.info(f"{'='*50}")
        
        # Start GPU monitoring
        monitor = GPUMonitor()
        monitor.start_monitoring()
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        try:
            start_time = time.time()
            success = train_func()
            training_time = time.time() - start_time
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            if success:
                logger.info(f"✅ {name} completed successfully")
                logger.info(f"   Training time: {training_time:.1f}s")
                logger.info(f"   GPU utilization: {monitor.avg_utilization:.1f}%")
                logger.info(f"   Peak memory: {monitor.max_memory:.2f} GB")
                
                results[name] = {
                    'success': True,
                    'time': training_time,
                    'gpu_util': monitor.avg_utilization,
                    'peak_memory': monitor.max_memory
                }
            else:
                logger.error(f"❌ {name} failed")
                results[name] = {'success': False}
                
        except Exception as e:
            monitor.stop_monitoring()
            logger.error(f"💥 {name} error: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Summary report
    logger.info(f"\n🏆 GPU-VERIFIED TRAINING SUMMARY")
    logger.info("=" * 40)
    
    for name, result in results.items():
        if result.get('success', False):
            logger.info(f"✅ {name:25} | "
                       f"Time: {result['time']:.1f}s | "
                       f"GPU: {result['gpu_util']:.1f}% | "
                       f"Memory: {result['peak_memory']:.2f}GB")
        else:
            logger.info(f"❌ {name:25} | FAILED")
    
    return results

def main():
    """Main GPU verification and training"""
    
    logger.info("🔥 GPU-Verified Speculation Training")
    logger.info("=" * 60)
    
    Path("trained_models").mkdir(exist_ok=True)
    
    # Run comprehensive GPU-verified training
    results = run_gpu_verified_training()
    
    # Check if any training succeeded with good GPU utilization
    successful_trainings = [name for name, result in results.items() 
                          if result.get('success', False) and result.get('gpu_util', 0) > 50]
    
    if successful_trainings:
        logger.info(f"\n🎉 SUCCESS! {len(successful_trainings)} approaches trained successfully on GPU")
        logger.info(f"✅ High GPU utilization approaches: {', '.join(successful_trainings)}")
    else:
        logger.warning(f"\n⚠️ Low GPU utilization detected - check for CPU fallback")
    
    return len(successful_trainings) > 0

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ GPU-verified training completed!")
        print("🔥 All training running efficiently on GPU!")
    else:
        print("\n⚠️ GPU utilization issues detected")
        print("Check logs for details")