"""
Setup script for cs336-llm-system.

Install with:  pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="cs336-llm-system",
    version="0.1.0",
    description="CS336 Chinese Edition: LLM System Learning Roadmap",
    author="CS336 Contributors",
    python_requires=">=3.10",
    packages=find_packages(include=["src*", "labs*", "project*"]),
    install_requires=[
        "torch>=2.0.0",
        "einops",
        "tiktoken",
        "numpy",
        "matplotlib",
        "scipy",
    ],
    extras_require={
        "triton": ["triton>=2.0.0; platform_system == 'Linux'"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
