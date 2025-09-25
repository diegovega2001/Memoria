"""
Setup script para CompCars Analysis.

NOTA: Este proyecto usa pyproject.toml para la configuración moderna.
Este setup.py se mantiene por compatibilidad con herramientas legacy.
Para instalación moderna usar: pip install -e .
"""

from setuptools import setup, find_packages

# Leer el README para la descripción larga
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Leer requirements básicos desde requirements-dev.txt
def read_requirements():
    """Leer requirements básicos para compatibilidad."""
    try:
        with open("requirements-dev.txt", "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "torch>=2.2.0",
            "torchvision>=0.17.0", 
            "numpy>=1.26.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.7.0",
            "pillow>=10.0.0",
        ]

setup(
    name="compcars-analysis",
    version="0.1.0",
    author="Diego Vega",
    author_email="diegovega2000@live.com",
    description="Análisis de vehículos con deep learning usando dataset CompCars",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diegovega2001/Memoria",
    packages=find_packages(where=".", include=["src*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12",
    install_requires=read_requirements(),
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.0.0"],
        "jupyter": ["jupyter>=1.0.0", "ipywidgets>=8.0.0"],
    },
    entry_points={
        "console_scripts": [
            "compcars-train=src.pipeline.FineTuningPipeline:main",
            "compcars-analyze=src.pipeline.EmbeddingsPipeline:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
