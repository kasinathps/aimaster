import setuptools


setuptools.setup(
  name='aimaster',
  url='https://github.com/kasinathps/aimaster',
  author='Kasinath P.S',
  author_email='kasinathps@gmail.com',
  packages=['aimaster','aimaster.nn1hlib','aimaster.nn1hlnb'],
  install_requires=['numpy','scipy'],
  version='1.0.1',
  license='MIT',
  description='Artificial Neural Network learning tools',
  long_description=open('README.md').read(),
)