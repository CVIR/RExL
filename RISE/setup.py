from setuptools import setup, find_packages

setup(name='RISE',
      version='1',
      install_requires=['scikit-image', 'matplotlib', 'torch', 'torchvision'],
      packages=find_packages()
)
