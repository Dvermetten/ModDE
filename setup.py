"""Setup script."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = "0.0.4"

setuptools.setup(
    name='modde',
    version=__version__,
    author="Diederick Vermetten",
    author_email="d.l.vermetten@liacs.leidenuniv.nl",
    description="Package Containing Modular DE optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'ioh'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)