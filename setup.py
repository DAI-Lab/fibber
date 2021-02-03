#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', encoding='utf-8') as history_file:
    history = history_file.read()

install_requires = [
    'numpy>=1.18.0',
    'tensorflow-gpu>=2.0.0',
    'tensorflow-hub>=0.9.0',
    'torch<2,>=1.0',
    'torchvision<1,>=0.4.2',
    'transformers>=2.4.0',
    'tqdm>=4.0.0',
    'spacy>=2.0.0',
    'pandas>=1.0.0',
    'nltk>=3.0',
    'stanza>=1.1.0',
    'sentence-transformers>=0.3.0'
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
]

development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r2>=0.2.5,<0.3',
    'nbsphinx>=0.5.0,<0.7',
    'Sphinx==3.2.1',
    'pydata-sphinx-theme',
    'autodocsumm>=0.1.10',
    'PyYaml>=5.3.1,<6',
    'argh>=0.26.2,<1',
    'sphinx_rtd_theme>=0.4,<1',
    'ipython>=7,<8',

    # style check
    'flake8>=3.7.7',
    'isort>=5',

    # fix style issues
    'autoflake>=1.2',
    'autopep8>=1.4.3',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1',
    'tox>=2.9.1',
]

setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description='Fibber is a benchmarking suite for adversarial attacks on text classification.',
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    install_package_data=True,
    install_requires=install_requires,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='fibber fibber fibber',
    name='fibber',
    packages=find_packages(include=['fibber', 'fibber.*']),
    python_requires='>=3.6',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/DAI-Lab/fibber',
    version='0.2.2',
    zip_safe=False,
)
