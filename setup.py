"""
Setup script for Enhanced Pre-gated MoE
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="enhanced-pregated-moe",
    version="0.1.0",
    author="Enhanced Pre-gated MoE Team",
    author_email="your-email@example.com",
    description="Enhanced Pre-gated MoE for RTX 3090: Advanced Speculation and Memory Optimization",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/specMoE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "benchmark": [
            "wandb>=0.16.0",
            "tensorboard>=2.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "specmoe-demo=main:main",
            "specmoe-pretrained=main_pretrained:main",
        ],
    },
    keywords="mixture-of-experts, moe, speculation, gpu-optimization, rtx3090, transformers",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/specMoE/issues",
        "Source": "https://github.com/yourusername/specMoE",
        "Documentation": "https://github.com/yourusername/specMoE/blob/main/README.md",
    },
)