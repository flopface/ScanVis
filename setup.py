from setuptools import setup, find_packages

setup(
    name="ScanVis",
    version="0.1",
    packages=find_packages(),
    description="A library for visualising MRI scans",
    author="Tom Melichar",
    author_email="tbmelichar@gmail.com",
    install_requires=['numpy', 'matplotlib', 'scipy', 'SimpleITK', 'ipywidgets', 'scikit.image'],
    python_requires=">=3.6",
)