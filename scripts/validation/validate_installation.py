#!/usr/bin/env python3
"""
Installation Validation Script

Validates that SpecMoE is properly installed and configured.
"""

import sys
import subprocess
import importlib
from pathlib import Path
import torch


def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_pytorch_installation():
    """Check PyTorch installation and CUDA availability."""
    print("\n🔥 Checking PyTorch installation...")
    
    try:
        import torch
        import torchvision
        print(f"   ✅ PyTorch {torch.__version__}")
        print(f"   ✅ TorchVision {torchvision.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA available ({gpu_count} GPU(s))")
            print(f"   ✅ Primary GPU: {gpu_name}")
            
            # Check memory
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   ✅ GPU Memory: {memory_gb:.1f} GB")
            
            return True, True  # pytorch_ok, cuda_ok
        else:
            print("   ⚠️  CUDA not available (CPU-only mode)")
            return True, False
            
    except ImportError as e:
        print(f"   ❌ PyTorch not found: {e}")
        return False, False


def check_specmoe_installation():
    """Check SpecMoE package installation."""
    print("\n⚡ Checking SpecMoE installation...")
    
    try:
        import src
        print("   ✅ SpecMoE src module found")
        
        # Check key modules
        key_modules = [
            "src.models",
            "src.evaluation", 
            "src.training",
            "src.utils"
        ]
        
        for module_name in key_modules:
            try:
                importlib.import_module(module_name)
                print(f"   ✅ {module_name}")
            except ImportError as e:
                print(f"   ❌ {module_name}: {e}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"   ❌ SpecMoE not found: {e}")
        print("   💡 Try: pip install -e .")
        return False


def check_dependencies():
    """Check required dependencies."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "numpy", "pandas", "matplotlib", "seaborn",
        "scikit-learn", "tqdm", "psutil", "yaml"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   💡 Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_directory_structure():
    """Check project directory structure."""
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        "src", "config", "results", "scripts", 
        "experiments", "docs"
    ]
    
    current_dir = Path.cwd()
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = current_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ (missing)")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print("\n   💡 Run from SpecMoE project root directory")
        return False
    
    return True


def check_configuration():
    """Check configuration files."""
    print("\n⚙️  Checking configuration...")
    
    try:
        from src.utils import ConfigManager
        config = ConfigManager()
        
        # Validate config
        config.validate_config()
        print("   ✅ Configuration valid")
        
        # Check available options
        architectures = config.get_available_architectures()
        strategies = config.get_available_strategies()
        hardware = config.get_available_hardware()
        
        print(f"   ✅ Architectures: {', '.join(architectures)}")
        print(f"   ✅ Strategies: {', '.join(strategies)}")
        print(f"   ✅ Hardware configs: {', '.join(hardware)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False


def run_quick_test():
    """Run a quick functionality test."""
    print("\n🧪 Running quick test...")
    
    try:
        from src.utils import ConfigManager
        
        # Test configuration loading
        config = ConfigManager()
        arch_config = config.get_architecture_config("switch_transformer")
        print(f"   ✅ Config loaded: Switch Transformer ({arch_config.num_experts} experts)")
        
        # Test model imports
        from src.models import pg_moe_strategy
        print("   ✅ Strategy import successful")
        
        # Test evaluation framework
        from src.evaluation import iso_cache_framework
        print("   ✅ Evaluation framework import successful")
        
        print("   ✅ Quick test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Quick test failed: {e}")
        return False


def print_system_info():
    """Print system information."""
    print("\n💻 System Information:")
    
    # Python info
    print(f"   Python: {sys.version}")
    print(f"   Platform: {sys.platform}")
    
    # PyTorch info
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    except ImportError:
        pass


def main():
    """Main validation function."""
    print("🚀 SpecMoE Installation Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch Installation", lambda: check_pytorch_installation()[0]),
        ("SpecMoE Installation", check_specmoe_installation),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Configuration", check_configuration),
        ("Quick Test", run_quick_test)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ {name} check failed with error: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 Validation Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! SpecMoE is ready to use.")
        print("\n📚 Next steps:")
        print("   - Run quick evaluation: python -m src.evaluation.run_evaluation --architecture switch_transformer --strategy intelligent --batch_sizes 1,8")
        print("   - View documentation: cat RUNNING_INSTRUCTIONS.md")
        print("   - Check examples: ls examples/")
    else:
        print(f"\n⚠️  {total - passed} checks failed. Please address the issues above.")
        print("\n💡 Common solutions:")
        print("   - Install missing dependencies: pip install -r requirements.txt")
        print("   - Install SpecMoE: pip install -e .")
        print("   - Run from project root directory")
    
    print_system_info()
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)