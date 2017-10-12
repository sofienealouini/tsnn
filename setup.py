from setuptools import setup, find_packages

setup(
    name='tsnn',
    version='0.0.17',
    author='Sofiene Alouini',
    author_email='sofiene.alouini@gmail.com',
    url='https://github.com/sofienealouini/tsnn',
    description='Time Series Neural Networks (Keras wrapper)',
    long_description=open('README.md').read(),
    packages=find_packages(),
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
        'numpy >= 1.13.0',
        'matplotlib >= 2.1.0',
        'pandas >= 0.20.0',
        'scikit-learn >= 0.19.0',
        'tensorflow >= 1.3.0',
        'keras >= 2.0.8'
      ]
)
