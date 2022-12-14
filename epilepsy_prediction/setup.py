"""Install clusterexplain package."""
from pathlib import Path

from setuptools import find_packages, setup

GIT_URL = "https://github.ibm.com/Yoav-Kan-Tor/public_epilepsy"
pkg_name = "epilepsy_prediction"




def get_version(filename):
    """Read version string from filename with string `__version__`"""
    with open(filename, encoding="utf-8") as f:
        for line in f.readlines():
            if line.startswith("__version__"):
                quotes_type = '"' if '"' in line else "'"
                version = line.split(quotes_type)[1]
                return version
    raise RuntimeError("Unable to find version string.")


setup(
    name=pkg_name,
    version=get_version(Path(pkg_name) / "__init__.py"),
    # packages=find_packages(exclude=['scripts', 'data', 'tests']),
    packages=find_packages(),
    description="epilepsy prediction profile creation by IBM Haifa Research Labs",
    long_description_content_type="text/markdown",
    url=GIT_URL,
    author="IBM Research Haifa Labs - Causal Machine Learning for Healthcare and Life Sciences",
    # author_email=None,
    license="Apache License 2.0",
    keywords="causal inference healthcare",
    install_requires=[
        "numpy",
        "pandas",
        "pyyaml",
        "matplotlib",
        "seaborn",
        "xgboost==1.6.1",
        "sklearn",
        "fuse-med-ml"
    ],
    extras_require={
    },
    include_package_data=True,
    package_data={pkg_name: [str(Path(pkg_name) / "tests" / "data" / "*.csv")]},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        # "License :: OSI Approved :: Apache Software License",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
)
