from setuptools import setup, find_packages

setup(
    name="libian_metrics",
    version="0.1.0",
    description="Python toolkit for quantifying ancient character glyph image quality metrics",
    author="LiShift Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "scikit-image>=0.19.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
        "pandas>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "libian_metrics=libian_metrics.cli:main",
        ],
    },
)
