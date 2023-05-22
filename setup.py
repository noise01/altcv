from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="altcv",
    version="0.0.1",
    install_requires=["opencv-python", "numpy", "torch", "torchvision"],
    description="",
    long_description=long_description,
    packages=find_packages(),
    license="MIT",
)
