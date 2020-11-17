import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="causal-lasso",
    version="1.0.0",
    author="Manon Romain",
    author_email="manon.romain@ens.fr",
    description="Causal Lasso",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manon643/causal_lasso",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
