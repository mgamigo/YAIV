import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yaiv", # Replace with your username
    version="0.0.1",
    author="Martin Gutierrez-Amigo",
    author_email="<martin00gutierrez@outlook.com>",
    description="Yet another Ab Initio Visualizer with a variety of tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mgamigo/YAIV",
    packages=setuptools.find_packages(),
    install_requires=[
        "ipympl==0.9.1"
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
