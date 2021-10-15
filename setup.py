import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME ="one_Neuron_pypi"
USERNAME = "LijiAlex"

setuptools.setup(
    name="ann_implementation",
    version="0.0.1",
    author=USERNAME,
    author_email="liji.alex@gmail.com",
    description="A small package for ANN Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LijiAlex/ANN_implimentation-DLCVNLP-demo",
    project_urls={
        "Bug Tracker": f"https://github.com/{USERNAME}/{PROJECT_NAME}/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages = ["src"],
    #packages = setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas"
    ]
)