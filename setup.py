#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentic-scent-analytics",
    version="1.0.0",
    author="Terragon Labs",
    author_email="contact@terragonlabs.com",
    description="LLM-powered industrial AI platform for smart factory e-nose deployments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terragonlabs/agentic-scent-analytics",
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
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "click>=8.0.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "sqlalchemy>=1.4.0",
        "redis>=3.5.0",
        "cryptography>=3.4.0",
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
        "industrial": [
            "psutil>=5.8.0",
            "scipy>=1.7.0",
            "prometheus-client>=0.12.0",
            "aioredis>=2.0.0",
            "asyncpg>=0.25.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.3.0",
            "httpx>=0.24.0",
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
            "agentic-scent=agentic_scent.cli:main",
            "quantum-planner=quantum_planner.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/terragonlabs/agentic-scent-analytics/issues",
        "Source": "https://github.com/terragonlabs/agentic-scent-analytics",
        "Documentation": "https://agentic-scent-analytics.readthedocs.io/",
    },
)