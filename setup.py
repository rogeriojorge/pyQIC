import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qicna",
    version="0.3.6",
    author="Rogerio Jorge and Matt Landreman",
    author_email="rogerio.jorge@tecnico.ulisboa.pt",
    description="Quasi-isodynamic Stellarator Construction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/rogeriojorge/pyQIC",
    #install_requires=['numpy', 'scipy'],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
#    packages=["qicna"],
