from setuptools import find_packages, setup

setup(
    name='topotest',
    packages=find_packages(),
    version='0.1.0',
    description='Onesample and twosample topological goodness-of-fit tests',
    author='Rafal Topolnicki',
    install_requires=['numpy', 'gudhi', 'scikit-learn', 'pandas', 'tqdm']
)
