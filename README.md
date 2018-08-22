# KNN Classifier and Impurity Measurements

This project attemps to implement a KNN classifier and some impurity measurement algorithms.

The KNN classifier follows the documentation of the [sci-kit learn](http://scikit-learn.org) KNN classifier. That is, some methods, classes and modules have similar signatures.

## Project structure

The project is structure as follows:

    src
    |
    |___classifiers
    |
    |___datasets
    |
    |___ext
    |
    |___metrics
    |
    |___profile
    |
    |___test
    |
    |___util

- classifiers: a package which contains classifiers

- datasets: a folder to store datasets

- ext: external code

- metrics: a package for impurity algorithms implementations

- profile: a folder for profiling tests

- test: unit test package

- util: util package

## Dependencies

- Python 3

- Numpy

- Pandas (optional, for better visualization in measures.py)

For some tests in jupyter notebooks:

- scikit learn

- tensorflow

## KNN Classifier

The implemented KNN classifier has the following features:

- Brute force algorithm

- KD tree algorithm for fast queries

- Prediction method

- K Neighbors queries

## Impuriy measurements

The metrics package currently provide the following impurity algorithm:

- Classification error

- Entropy

- Gini

## Getting started

See the __demo__ jupyter notebook to check the mainly features.

### Jupyter notebooks

- demo: overall usage demonstration

- diabetes: classifies the diabetes weka data set

- iris: classifies the iris weka data set

- vote: classifies the vote weka data set

- mnist: classifies the MNIST database

- mnist_param_tuning: attemp to find good hyper parameters in the KNN classifier

- feature_selection_mnist: use some algorithms to perform attributes selection on MNIST database

- pca_mnist: perform a principal component analysis(PCA) in MNIST database to reduce the dimensionality
