from setuptools import find_packages, setup

setup(
    name="rl_transformer",
    packages=find_packages(),
    url="https://github.com/qgallouedec/rl_transformer",
    description="RL Transformer",
    long_description=open("README.md").read(),
    install_requires=["gym==0.21", "torch", "black", "isort", "pytest", "stable-baselines3"],
)
