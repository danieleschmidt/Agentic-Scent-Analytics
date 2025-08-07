#!/usr/bin/env python3
"""
Setup script for Sentiment Analyzer Pro
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "fastapi>=0.104.1",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.5.0",
            "transformers>=4.35.2",
            "torch>=2.1.1",
            "vaderSentiment>=3.3.2",
            "textblob>=0.17.1",
            "click>=8.0.0",
        ]

setup(
    name="sentiment-analyzer-pro",
    version="1.0.0",
    author="Terragon Labs",
    author_email="info@terragonlabs.com",
    description="Advanced multi-model sentiment analysis platform with real-time processing and enterprise-grade security",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terragonlabs/sentiment-analyzer-pro",
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
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "isort>=5.12.0",
        ],
        "industrial": [
            "psycopg2-binary>=2.9.9",
            "redis>=5.0.1",
            "aioredis>=2.0.1",
            "asyncpg>=0.29.0",
        ],
        "llm": [
            "openai>=1.3.6",
            "anthropic>=0.7.7",
        ],
        "all": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "psycopg2-binary>=2.9.9",
            "redis>=5.0.1",
            "openai>=1.3.6",
            "anthropic>=0.7.7",
        ]
    },
    entry_points={
        "console_scripts": [
            "sentiment-analyzer=sentiment_analyzer.cli:cli",
        ],
    },
    include_package_data=True,
    keywords=[
        "sentiment-analysis",
        "nlp",
        "machine-learning", 
        "artificial-intelligence",
        "text-analysis",
        "emotion-detection",
        "transformers",
        "fastapi",
        "production-ready",
        "enterprise",
    ],
    project_urls={
        "Bug Reports": "https://github.com/terragonlabs/sentiment-analyzer-pro/issues",
        "Source": "https://github.com/terragonlabs/sentiment-analyzer-pro",
        "Documentation": "https://sentiment-analyzer-pro.readthedocs.io/",
    },
)