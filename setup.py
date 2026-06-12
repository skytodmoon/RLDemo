from setuptools import setup, find_packages

setup(
    name='industrial_temp_control',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'stable-baselines3',
        'numpy',
    ],
)