#from distutils.core import setup
from setuptools import setup
from pathlib import Path

with Path("requirements.txt").open() as f:
    requirements = list(f.readlines())

# Get the long description from the README file
with Path("README.md").open(encoding="utf-8") as f:
    long_description = f.read()

package_dir = "ignnition"
with (Path(package_dir) / "_version.py").open() as f:
    _vars = dict()
    exec(f.read(), _vars)
    version = _vars.get("__version__", "0.0.0")
    del _vars

setup(
  name='ignnition',         # How you named your package folder (MyLib)
  packages=['ignnition'],   # Choose the same as "name"
  version=version,      # Start with a small number and increase it with every change you make
  license='Apache license 2.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description='Library for fast prototyping of GNN',   # Give a short description about your library
  long_description=long_description,
  long_description_content_type="text/markdown",
  author='Barcelona Neural Networking Center',                   # Type in your name
  author_email='ignnition@contactus.net',      # Type in your E-Mail
  url='https://ignnition.net/',   # Provide either the link to your github or to your website
  package_data={'ignnition': ['schema.json']},
  keywords=['Machine Learning', 'Graph Neural Networks', 'Networking', 'Artificial Intelligence'],   # Keywords that define your package best
  install_requires=requirements,
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: Apache Software License',   # Again, pick a license
    'Programming Language :: Python :: 3.7',
  ],
)
