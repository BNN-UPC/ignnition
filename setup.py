#from distutils.core import setup
from setuptools import setup

setup(
  name = 'ignnition',         # How you named your package folder (MyLib)
  packages = ['ignnition'],   # Choose the same as "name"
  version = 'v_0.02',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Library for fast prototyping of GNN',   # Give a short description about your library
  author = 'Barcelona Neural Networking Center',                   # Type in your name
  author_email = 'ignnition@contactus.net',      # Type in your E-Mail
  url = 'https://ignnition.net/',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/knowledgedefinednetworking/ignnition/archive/v_0.02.tar.gz',    # I explain this later on
  package_data={'ignnition': ['schema.json']},
  keywords = ['Machine Learning', 'Graph Neural Networks', 'Networking', 'Artificial Intelligence'],   # Keywords that define your package best
  install_requires=[
          "validators",
          "beautifulsoup4",
          "absl-py==0.9.0",
          "astor==0.8.1",
          "attrs==19.3.0",
          "cachetools==4.1.0",
          "certifi==2020.4.5.1",
          "chardet==3.0.4",
          "gast==0.3.3",
          "google-auth==1.14.1",
          "google-auth-oauthlib==0.4.1",
          "google-pasta==0.2.0",
          "grpcio==1.28.1",
          "h5py==2.10.0",
          "idna==2.9",
          "importlib-metadata==1.6.0",
          "jsonschema==3.2.0",
          "Keras==2.4.3",
          "Keras-Applications==1.0.8",
          "Keras-Preprocessing==1.1.2",
          "Markdown==3.2.1",
          "numpy==1.18.3",
          "oauthlib==3.1.0",
          "opt-einsum==3.2.1",
          "protobuf==3.11.3",
          "pyasn1==0.4.8",
          "pyasn1-modules==0.2.8",
          "pyrsistent==0.16.0",
          "requests==2.23.0",
          "requests-oauthlib==1.3.0",
          "rsa==4.0",
          "scipy==1.4.1",
          "six==1.14.0",
          "tensorboard==2.3.0",
          "tensorboard-plugin-wit==1.7.0",
          "tensorboardX==1.9",
          "tensorflow==2.3.0",
          "tensorflow-estimator==2.3.0",
          "termcolor==1.1.0",
          "urllib3==1.25.9",
          "Werkzeug==1.0.1",
          "wrapt==1.12.1",
          "zipp==3.1.0",
          "networkx==2.4"
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.8.5',
  ],
)
