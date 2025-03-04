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

with open(path.join(script_directory, 'README.md')) as f:
    long_description = f.read()

requirements = list()
with open(path.join(script_directory, 'requirements.txt')) as f:
    for line in f.readlines():
        line = line.strip()
        if line:
            if not line.startswith("#"):
                requirements.append(line)

setup(
    name=package_name,
    python_requires='>=3.6',
    version=version,
    description='Hierarchical niche space analysis in python',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/jolespin/nichespace',

    # Author details
    author='Josh L. Espinoza',

    # Choose your license
    license='GPLv3',
    # provides=['metabolic_niche_space'],
    # packages=['metabolic_niche_space'],
    # py_modules=["manifold", "neighbors"],
    packages=find_packages(),
    
    install_requires=requirements, #[:-1],
    tests_require=requirements, #[-1:]
     scripts=[
     "bin/edgelist_to_clusters.py",
     ],
)