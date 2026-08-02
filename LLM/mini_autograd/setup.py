from setuptools import find_packages, setup

setup(
    name="mini_autograd",
    version="0.1.0",
    description="一个从零实现的微型自动微分引擎，用于学习 PyTorch 内部原理",
    packages=find_packages(),
    install_requires=["numpy>=1.24"],
    python_requires=">=3.9",
)
