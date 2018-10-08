from setuptools import setup

setup(
  name='aimaster',
  url='https://github.com/kasinathps/aimaster',
  author='Kasinath P.S',
  author_email='kasinathps@gmail.com',
  packages=['aimaster'],
  install_requires=['numpy','scipy'],
  license='MIT',
  description='Artificial Neural Network learning tools',
  long_description=open('README.md').read(),
)