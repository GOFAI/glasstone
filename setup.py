from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='glasstone',
    version='0.0.1',
    description='Python library for modelling nuclear weapons effects',
    long_description=readme,
    author='Edward Geist',
    author_email='egeist@stanford.edu',
    url='https://github.com/GOFAI/glasstone',
    license='MIT',
    packages=['glasstone'],
    install_requires=['numpy', 'scipy', 'affine'])
