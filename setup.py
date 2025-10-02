"""
Setup script for MHA-DQN Portfolio Optimization Research Project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mha-dqn-portfolio-research",
    version="1.0.0",
    author="Zelalem Abahana",
    author_email="zga5029@psu.edu",
    description="Multi-Head Attention Deep Q-Networks for Portfolio Optimization: A Sentiment-Integrated Reinforcement Learning Approach",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zabahana/mha-dqn-portfolio-research",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "collect-data=scripts.collect_data:main",
            "train-mha-dqn=scripts.train_mha_dqn:main",
            "evaluate-model=scripts.evaluate_model:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
