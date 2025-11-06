from setuptools import setup, find_packages

setup(
    name="intelligent-nesting",
    version="0.1.0",
    description="Sistema Inteligente de Nesting com Deep RL",
    author="Seu Nome",
    author_email="seu.email@universidade.br",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "shapely>=2.0.0",
        "gymnasium>=0.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "viz": [
            "plotly>=5.14.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nesting-train=experiments.04_rl_training:main",
            "nesting-eval=experiments.05_evaluation:main",
        ],
    },
)