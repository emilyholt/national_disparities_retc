from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    install_requires=[
        "black",
        "geopandas",
        "jupyter",
        "jupyter-black",
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "scipy",
        "six",
        "scikit-learn",
        "statsmodels",
        "tabulate",
        "tqdm",
        "pytz",
    ]
)
