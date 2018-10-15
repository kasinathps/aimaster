from setuptools import setup

setup(
  name='aimaster',
  url='https://github.com/kasinathps/aimaster',
  author='Kasinath P.S',
  author_email='kasinathps@gmail.com',
  packages=['aimaster','aimaster.nn1hlnb','aimaster.nn1hlib'],
  install_requires=['numpy','scipy'],
  version='1.0',
  license='MIT',
  description='Artificial Neural Network learning tools',
  long_description=open('README.md').read(),
)