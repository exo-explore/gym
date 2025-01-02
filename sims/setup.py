from setuptools import setup, find_packages

setup(
    name="diloco_sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "torchtext",
        "portalocker",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "Pillow",
        "wandb",
        "transformers",
        "datasets",
    ],
)
