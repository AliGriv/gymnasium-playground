from setuptools import setup, find_packages
import os
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='gym-playground',
    version='0.3.1',
    author='Ali Grivani',
    description='A package for running and testing various gym environments.',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'gym-playground=gym_playground.cli:cli',
        ],
    },
)
