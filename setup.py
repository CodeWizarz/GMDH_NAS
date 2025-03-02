from setuptools import setup, find_packages

setup(
    name="auto_gmdh",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "bayesian-optimization"
    ],
    author="Cathugger",
    description="Auto-GMDH: An automated neural network using GMDH and NAS.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CodeWizarz/AUTO_GMDH",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
