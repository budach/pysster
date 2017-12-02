from setuptools import setup

setup(
    name = 'pysster',
    version = '1.0.1',
    description = 'A DNA/RNA Sequence/STructure ClassifiER',
    url = 'https://github.com/budach/pysster',
    author = 'Stefan Budach',
    author_email = 'budach@molgen.mpg.de',
    license = 'MIT',
    install_requires =  [
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'keras>=2.1.1',
        'tensorflow>=1.4.0',
        'h5py',
        'logging_exceptions',
        'Pillow',
        'forgi'
    ],
    packages = ['pysster'],
    python_requires = '>=3.5',
    include_package_data = True,
    zip_safe = False
)
