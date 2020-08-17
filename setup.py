import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bempp-cavity-sheeshee", # Replace with your own username
    version="0.0.1",
    author="Samuel Sheehy",
    author_email="samuelsheehy95@gmail.com",
    description="An implementation of BEMPP for problems with nested domains.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sheeshee/MSc-Project-Code",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
