#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum-task-planner",
    version="1.0.0",
    author="Terragon Labs",
    author_email="contact@terragonlabs.com",
    description="Quantum-inspired task planning and optimization system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terragonlabs/quantum-task-planner",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "click>=8.0.0",
        "asyncio>=3.4.3",
        "typing-extensions>=4.0.0",
        "dataclasses>=0.8;python_version<'3.7'",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "coverage>=6.0.0",
            "pytest-cov>=4.0.0",
        ],
        "performance": [
            "psutil>=5.8.0",
            "scipy>=1.7.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "graphviz>=0.20.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "coverage>=6.0.0",
            "pytest-cov>=4.0.0",
            "psutil>=5.8.0",
            "scipy>=1.7.0",
            "matplotlib>=3.5.0",
            "graphviz>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantum-planner=quantum_planner.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/terragonlabs/quantum-task-planner/issues",
        "Source": "https://github.com/terragonlabs/quantum-task-planner",
        "Documentation": "https://quantum-task-planner.readthedocs.io/",
    },
)