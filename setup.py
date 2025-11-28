from setuptools import setup, find_packages

setup(
    name='nob-eng-translation',
    version='1.0.0',
    description='Norwegian-English Neural Machine Translation with LoRA Fine-tuning',
    author='Your Name',
    email='your.email@example.com',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        line.strip() for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)