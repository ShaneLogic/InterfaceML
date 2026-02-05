"""
Setup script for InterfaceML package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="interfaceml",
    version="1.0.0",
    author="Interface Modeling Lab",
    author_email="interface@example.com",
    description="A professional toolkit for heterojunction modeling and DFT calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/InterfaceML",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pymatgen>=2022.0.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "werkzeug>=2.0.0",
        "matplotlib>=3.6.0",
    ],
    extras_require={
        "perf": [
            "scipy>=1.7.0",
        ],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "interfaceml-build=interfaceml.cli.build_interface:main",
            "interfaceml-fix=interfaceml.cli.fix_layers:main",
            "interfaceml-web=interfaceml.web.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "interfaceml.web": [
            "static/**/*",
            "templates/**/*",
        ],
    },
    zip_safe=False,
)
