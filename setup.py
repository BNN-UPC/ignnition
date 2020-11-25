from distutils.core import setup

setup(
  name = 'ignnition',         # How you named your package folder (MyLib)
  packages = ['ignnition'],   # Choose the same as "name"
  version = '0.01',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Library for fast prototyping of GNN',   # Give a short description about your library
  author = 'Barcelona Neural Networking Center',                   # Type in your name
  author_email = 'david.pujolperich@estudiantat.upc.edu',      # Type in your E-Mail
  url = 'https://github.com/dpujol14/ignnition',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/dpujol14/ignnition/archive/0.01.tar.gz',    # I explain this later on
  package_data={
      'ignnition': ['ignnition/*.json'],
   },
  keywords = ['ML', 'GNN', 'NETWORKING'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'validators',
          'beautifulsoup4',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.7',
  ],
)
