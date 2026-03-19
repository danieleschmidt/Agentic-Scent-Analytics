from setuptools import setup, find_packages

setup(
    name="agentic-scent-analytics",
    version="1.0.0",
    description="Agentic framework for e-nose chemical sensor data analysis and odorant classification",
    author="Daniel Schmidt",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24"],
)
