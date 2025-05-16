from setuptools import setup, find_packages

# Core dependencies (keep minimal for setup.py)
install_requires = [
    'numpy>=1.23.4',  # Using >= to be more flexible, but specify a minimum
    'sentencepiece>=0.1.99',
    'pyyaml>=5.4',
    'tqdm>=4.55.0',
    'loguru>=0.5.3',
    'nltk>=3.4.4',
    'requests>=2.25.1',
    'scikit-learn>=0.24.0',
    'fuzzywuzzy>=0.18.0',
    'tensorboard>=2.4.1',
    'rouge-score>=0.1.2',
    'transformers>=4.40.0',
    'editdistance>=0.8.1',
    'peft>=0.10.0',
    'torch', # No version here, see note below
    'torch_geometric', #No version here, see note below
]

# Conditional dependency for Python versions less than 3.7
if __import__('sys').version_info < (3, 7):
    install_requires.append('dataclasses~=0.7')

# Optional dependencies (moved to extras_require)
extras_require = {
    'full': [ # Installs all, including those needed for examples, dev, etc.
        'aiohttp>=3.11.16',
        'fasttext>=0.9.2',
        'pkuseg>=0.0.25',
        'aiohappyeyeballs>=2.6.1',
        'async-timeout>=5.0.1',
        'attrs>=25.3.0',
        'certifi>=2025.1.31',
        'charset-normalizer>=3.4.1',
        'click>=8.1.8',
        'filelock>=3.18.0',
        'frozenlist>=1.5.0',
        'fsspec>=2025.3.2',
        'grpcio>=1.71.0',
        'huggingface-hub>=0.30.2',
        'idna>=3.10',
        'Jinja2>=3.1.6',
        'joblib>=1.4.2',
        'Markdown>=3.8',
        'MarkupSafe>=3.0.2',
        'mpmath>=1.13.0',
        'multidict>=6.4.3',
        'networkx>=3.4.2',
        'packaging>=24.2',
        'propcache>=0.3.1',
        'protobuf>=6.30.2',
        'psutil>=7.0.0',
        'pybind11>=2.13.6',
        'pyparsing>=3.2.3',
        'scipy>=1.15.2',
        'tokenizers>=0.21.1',
        'typing_extensions>=4.13.2',
        'urllib3>=2.4.0',
        'Werkzeug>=3.1.3',
        'yarl>=1.19.0'
    ],
    'fasttext': ['fasttext>=0.9.2'],  # Keep this, in case someone only wants fasttext
    'pkuseg': ['pkuseg>=0.0.25'], # Keep this
}

classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Human Machine Interfaces",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='crslab',
    version='0.1.1',  # Make sure this matches crslab/__init__.py
    author='CRSLabTeam',
    author_email='francis_kun_zhou@163.com',
    description='An Open-Source Toolkit for Building Conversational Recommender System(CRS)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/RUCAIBox/CRSLab',
    packages=find_packages(where='.', include=['crslab*']), #Include crslab packages
    classifiers=classifiers,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires='>=3.10',
    # Removed setup_requires, not generally recommended
    # No longer include -f in setup.py
)
