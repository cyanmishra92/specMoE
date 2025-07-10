#!/usr/bin/env python3
"""
Check Current Status of Data Collection and Models
Quick status checker for ongoing tasks
"""

import os
from pathlib import Path
import pickle
import json
import time
import subprocess

def check_file_status():
    """Check status of key files"""
    
    files_to_check = [
        "routing_data/proper_traces.pkl",
        "routing_data/diverse_traces.pkl", 
        "routing_data/massive_traces_50k.pkl",
        "trained_models/simple_speculation_model.pt",
        "trained_models/robust_speculation_model.pt"
    ]
    
    print("📁 FILE STATUS:")
    print("=" * 50)
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024*1024)
            mod_time = time.ctime(path.stat().st_mtime)
            print(f"✅ {file_path}")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   Modified: {mod_time}")
            
            # Try to read trace files and show sample info
            if file_path.endswith('.pkl') and 'traces' in file_path:
                try:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    print(f"   Traces: {len(data)} samples")
                    
                    # Check diversity if it's trace data
                    if len(data) > 0 and hasattr(data[0], 'get'):
                        # It's dict format
                        sample = data[0]
                        if 'target_routing' in sample:
                            routing = sample['target_routing']
                            expert_used = routing.argmax() if hasattr(routing, 'argmax') else "unknown"
                            print(f"   Sample expert: {expert_used}")
                    
                except Exception as e:
                    print(f"   Error reading: {e}")
            
        else:
            print(f"❌ {file_path} - NOT FOUND")
        print()

def check_running_processes():
    """Check for running Python processes"""
    
    print("🔄 RUNNING PROCESSES:")
    print("=" * 50)
    
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        python_processes = []
        for line in lines:
            if 'python' in line and ('collect_' in line or 'train_' in line):
                python_processes.append(line)
        
        if python_processes:
            for proc in python_processes:
                print(f"🐍 {proc}")
        else:
            print("No relevant Python processes running")
            
    except Exception as e:
        print(f"Error checking processes: {e}")
    
    print()

def check_gpu_usage():
    """Check GPU memory usage"""
    
    print("🖥️  GPU STATUS:")
    print("=" * 50)
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            memory_info = result.stdout.strip().split(',')
            used_mb = int(memory_info[0])
            total_mb = int(memory_info[1])
            usage_pct = (used_mb / total_mb) * 100
            
            print(f"GPU Memory: {used_mb} MB / {total_mb} MB ({usage_pct:.1f}%)")
            
            if usage_pct > 50:
                print("⚠️  High GPU usage - model training likely running")
            elif usage_pct > 10:
                print("✅ Moderate GPU usage - light processing")
            else:
                print("💤 Low GPU usage - likely idle")
        else:
            print("Unable to get GPU info")
            
    except Exception as e:
        print(f"Error checking GPU: {e}")
    
    print()

def check_diverse_trace_progress():
    """Check if diverse trace collection shows progress"""
    
    print("🎯 DIVERSE TRACE COLLECTION STATUS:")
    print("=" * 50)
    
    # Check log files
    log_files = ['diverse_traces.log', 'collect_diverse.log', 'nohup.out']
    
    for log_file in log_files:
        if Path(log_file).exists():
            print(f"📄 Found log: {log_file}")
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        print(f"   Lines: {len(lines)}")
                        if len(lines) > 5:
                            print("   Last 3 lines:")
                            for line in lines[-3:]:
                                print(f"   {line.strip()}")
            except Exception as e:
                print(f"   Error reading log: {e}")
            print()
    
    # Check for partial trace files
    routing_dir = Path("routing_data")
    if routing_dir.exists():
        partial_files = list(routing_dir.glob("*partial*")) + list(routing_dir.glob("*intermediate*"))
        if partial_files:
            print("🔄 Partial files found:")
            for f in partial_files:
                size_mb = f.stat().st_size / (1024*1024)
                print(f"   {f.name}: {size_mb:.2f} MB")
        else:
            print("No partial files found")

def main():
    """Main status check"""
    
    print("🚀 SPECULATIVE MOE STATUS CHECK")
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    check_file_status()
    check_running_processes() 
    check_gpu_usage()
    check_diverse_trace_progress()
    
    print("=" * 60)
    print("✅ Status check complete!")

if __name__ == "__main__":
    main()