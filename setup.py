from setuptools import setup, find_packages
from os import path

script_directory = path.abspath(path.dirname(__file__))

package_name = "nichespace"
version = None
with open(path.join(script_directory, package_name, '__init__.py')) as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
assert version is not None, f"Check version in {package_name}/__init__.py"

with open(path.join(script_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

requirements = []
with open(path.join(script_directory, 'requirements.txt')) as f:
    for line in f.readlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="niche-space",
    python_requires='>=3.6',
    version=version,
    description='Hierarchical niche space analysis in python',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Ensures README.md renders on PyPI
    url='https://github.com/jolespin/niche-space',
    author='Josh L. Espinoza',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    tests_require=requirements,
    scripts=[
        "bin/edgelist_to_clusters.py",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

