from setuptools import setup, find_packages

setup(
    name="gmdh_nas",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "matplotlib",
        "tqdm"
    ],
)
