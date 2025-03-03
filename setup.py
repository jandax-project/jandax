from setuptools import find_packages, setup

setup(
    name="jandax",
    version="0.1.0",
    description="Traceable and Portable DataFrames for C++ Integration",
    author="Jandax Team",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.8",
)
