"""
Configuration Management for SpecMoE

Provides centralized configuration loading and management across all components.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class ArchitectureConfig:
    """Configuration for MoE architecture."""
    num_experts: int
    num_layers: int
    routing_type: str
    expert_size_mb: float
    default_cache_size_mb: float


@dataclass 
class StrategyConfig:
    """Configuration for expert prefetching strategy."""
    description: str
    cache_enabled: bool
    **kwargs: Any  # Additional strategy-specific parameters


@dataclass
class HardwareConfig:
    """Configuration for hardware cost modeling."""
    memory_gb: int
    bandwidth_gb_s: int
    pcie_bandwidth_gb_s: int
    transfer_cost_ms_gb: float


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    default_batch_sizes: list
    statistical_runs: int
    confidence_interval: float
    iso_cache: Dict[str, float]
    metrics: list


@dataclass
class TrainingConfig:
    """Configuration for neural predictor training."""
    neural_predictor: Dict[str, Any]


class ConfigManager:
    """Centralized configuration manager for SpecMoE."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path or self._find_default_config()
        self.config = self._load_config()
        
    def _find_default_config(self) -> Path:
        """Find default configuration file."""
        # Look for config relative to this file
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        
        config_paths = [
            project_root / "config" / "default.yaml",
            current_dir / "config" / "default.yaml",
            Path("config/default.yaml"),
            Path("default.yaml")
        ]
        
        for path in config_paths:
            if path.exists():
                return path
                
        raise FileNotFoundError(
            f"Could not find default configuration file. Searched: {config_paths}"
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"Error loading config from {self.config_path}: {e}")
    
    def get_architecture_config(self, architecture: str) -> ArchitectureConfig:
        """Get architecture-specific configuration."""
        if architecture not in self.config.get('architectures', {}):
            available = list(self.config.get('architectures', {}).keys())
            raise ValueError(
                f"Unknown architecture '{architecture}'. Available: {available}"
            )
        
        arch_config = self.config['architectures'][architecture]
        return ArchitectureConfig(**arch_config)
    
    def get_strategy_config(self, strategy: str) -> StrategyConfig:
        """Get strategy-specific configuration."""
        if strategy not in self.config.get('strategies', {}):
            available = list(self.config.get('strategies', {}).keys())
            raise ValueError(
                f"Unknown strategy '{strategy}'. Available: {available}"
            )
        
        strategy_config = self.config['strategies'][strategy]
        return StrategyConfig(**strategy_config)
    
    def get_hardware_config(self, hardware: str) -> HardwareConfig:
        """Get hardware-specific configuration."""
        if hardware not in self.config.get('hardware', {}):
            available = list(self.config.get('hardware', {}).keys())
            raise ValueError(
                f"Unknown hardware '{hardware}'. Available: {available}"
            )
        
        hardware_config = self.config['hardware'][hardware]
        return HardwareConfig(**hardware_config)
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        eval_config = self.config.get('evaluation', {})
        return EvaluationConfig(**eval_config)
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        training_config = self.config.get('training', {})
        return TrainingConfig(**training_config)
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update_setting(self, key: str, value: Any) -> None:
        """Update a specific setting value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file."""
        output_path = output_path or self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def get_available_architectures(self) -> list:
        """Get list of available architectures."""
        return list(self.config.get('architectures', {}).keys())
    
    def get_available_strategies(self) -> list:
        """Get list of available strategies."""
        return list(self.config.get('strategies', {}).keys())
    
    def get_available_hardware(self) -> list:
        """Get list of available hardware configurations."""
        return list(self.config.get('hardware', {}).keys())
    
    def validate_config(self) -> bool:
        """Validate configuration structure and values."""
        required_sections = ['architectures', 'strategies', 'hardware', 'evaluation']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate architectures have required fields
        for arch, config in self.config['architectures'].items():
            required_fields = ['num_experts', 'num_layers', 'routing_type', 'expert_size_mb']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Architecture {arch} missing required field: {field}")
        
        # Validate strategies have required fields
        for strategy, config in self.config['strategies'].items():
            required_fields = ['description', 'cache_enabled']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Strategy {strategy} missing required field: {field}")
        
        return True
    
    def create_runtime_config(
        self, 
        architecture: str,
        strategy: str,
        hardware: str,
        **overrides
    ) -> Dict[str, Any]:
        """Create runtime configuration combining multiple config sections."""
        runtime_config = {
            'architecture': asdict(self.get_architecture_config(architecture)),
            'strategy': asdict(self.get_strategy_config(strategy)),
            'hardware': asdict(self.get_hardware_config(hardware)),
            'evaluation': asdict(self.get_evaluation_config()),
        }
        
        # Apply any overrides
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested keys like 'evaluation.statistical_runs'
                keys = key.split('.')
                config = runtime_config
                for k in keys[:-1]:
                    if k not in config:
                        config[k] = {}
                    config = config[k]
                config[keys[-1]] = value
            else:
                runtime_config[key] = value
        
        return runtime_config


# Global configuration instance
_config_manager = None

def get_config_manager(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> Dict[str, Any]:
    """Get current configuration dictionary."""
    return get_config_manager().config


def get_architecture_config(architecture: str) -> ArchitectureConfig:
    """Get architecture configuration."""
    return get_config_manager().get_architecture_config(architecture)


def get_strategy_config(strategy: str) -> StrategyConfig:
    """Get strategy configuration."""
    return get_config_manager().get_strategy_config(strategy)


def get_hardware_config(hardware: str) -> HardwareConfig:
    """Get hardware configuration."""
    return get_config_manager().get_hardware_config(hardware)


if __name__ == "__main__":
    # Example usage
    config_mgr = ConfigManager()
    
    # Validate configuration
    config_mgr.validate_config()
    print("✅ Configuration validation passed")
    
    # Show available options
    print("Available architectures:", config_mgr.get_available_architectures())
    print("Available strategies:", config_mgr.get_available_strategies())
    print("Available hardware:", config_mgr.get_available_hardware())
    
    # Example runtime configuration
    runtime_config = config_mgr.create_runtime_config(
        architecture="switch_transformer",
        strategy="intelligent",
        hardware="rtx_4090",
        batch_size=32
    )
    
    print("Runtime configuration created successfully")