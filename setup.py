from setuptools import setup, find_packages


setup(
    name='synpg',
    version='1.0.0',
    url='https://github.com/chrisonntag/synpg.git',
    author='Christoph Sonntag',
    author_email='author@gmail.com',
    description='A package for syntactic paraphase generation',
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'torch==1.2.0',
        'nltk==3.4.5',
        'numpy==1.18.1',
        'tqdm==4.26.0',
        'ipdb==0.12.3',
        'h5py==2.10.0'
    ],
    package_data={
        '': ['data/*', 'demo/*', 'scripts/*', 'subwordnmt/*'],
    }
)
