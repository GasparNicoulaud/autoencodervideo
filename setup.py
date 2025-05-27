from setuptools import setup, find_packages
import os

setup(
    name="autoencodervideo",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        line.strip() 
        for line in open('requirements.txt').readlines() 
        if line.strip() and not line.startswith('#')
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="Video autoencoder with programmable latent space control",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)