#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentic-scent-analytics",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.com",
    description="LLM-powered analytics platform for smart factory e-nose deployments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terragonlabs/agentic-scent-analytics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "asyncio-mqtt>=0.11.0",
        "aiohttp>=3.8.0",
        "pydantic>=1.9.0",
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "sqlalchemy>=1.4.0",
        "alembic>=1.7.0",
        "redis>=4.0.0",
        "prometheus-client>=0.12.0",
    ],
    extras_require={
        "industrial": [
            "opcua>=0.98.13",
            "modbus-tk>=1.1.2",
            "pyserial>=3.5",
        ],
        "llm": [
            "langchain>=0.1.0",
            "transformers>=4.20.0",
            "torch>=1.12.0",
        ],
        "dev": [
            "pytest>=6.2.0",
            "pytest-asyncio>=0.19.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "coverage>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agentic-scent=agentic_scent.cli:main",
        ],
    },
)