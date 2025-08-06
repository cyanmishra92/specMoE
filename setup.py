#!/usr/bin/env python
"""
Setup script for SpecMoE: Expert Prefetching for Mixture-of-Experts Models
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="specmoe",
    version="1.0.0",
    author="SpecMoE Research Team",
    description="Expert Prefetching for Mixture-of-Experts Models: Neural Prediction and Batch-Aware Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "pre-commit>=2.20.0",
        ],
        "jupyter": [
            "jupyter",
            "ipykernel",
            "nbformat",
        ],
        "wandb": [
            "wandb",
        ],
    },
    entry_points={
        "console_scripts": [
            "specmoe-train=src.training.train:main",
            "specmoe-eval=src.evaluation.evaluate:main",
            "specmoe-analyze=src.utils.analyze:main",
        ],
    },
    project_urls={
        "Documentation": "https://github.com/research/specMoE/docs",
        "Source": "https://github.com/research/specMoE",
        "Tracker": "https://github.com/research/specMoE/issues",
    },
)