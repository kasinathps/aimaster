import setuptools


setuptools.setup(
  name='aimaster',
  url='https://github.com/kasinathps/aimaster',
  author='Kasinath P.S',
  author_email='kasinathps@gmail.com',
  packages=setuptools.find_packages(),
  install_requires=['numpy', 'scipy', 'matplotlib'],
  version='2.4.8',
  license='MIT',
  description='Artificial Neural Network learning tools',
  long_description=open('README.md').read(),
)
