from setuptools import setup

setup(
    name='tsnn',
    version='0.0.2',
    author='Sofiene Alouini',
    author_email='sofiene.alouini@gmail.com',
    url='https://github.com/sofienealouini/tsnn',
    description='Time Series Neural Networks wrapper',
    long_description=open('README.md').read(),
    packages=['tsnn'],
    test_suite='tests',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Free for non-commercial use',
        'Operating System :: OS Independent',
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'keras'
      ]
)
