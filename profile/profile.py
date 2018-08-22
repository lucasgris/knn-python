from util.dataparser import DataParser as dp
from classifiers.neighbors import KNeighborsClassifier
from metrics.measurescores import ImpurityMeasures
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import numpy as np


if __name__ == '__main__':
    with PyCallGraph(output=GraphvizOutput()):
        X, y = dp.arff_data('datasets/iris.arff')
        classifier = KNeighborsClassifier(algorithm='kd_tree')
        classifier.fit(X, y)
        pred = classifier.predict(X)
