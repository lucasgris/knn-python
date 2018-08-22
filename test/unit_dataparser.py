import unittest
import numpy as np
import sklearn.datasets as datasets
from numpy.testing import assert_array_equal
from util.dataparser import DataParser as DP


class TestDataParser(unittest.TestCase):

    def test_parse(self):
        ids, X, classes = DP.parse('test_data.txt')

        assert_array_equal(ids, np.asarray([0, 1, 2]))
        assert_array_equal(X, np.asarray([[5.1, 3.5, 1.4, 0.2],
                                          [7.0, 3.2, 4.7, 1.4],
                                          [6.3, 3.3, 6.0, 2.5]]))
        assert_array_equal(classes, np.asarray(['setosa', 'versicolor',
                                                'virginica']))

    def test_parse_arff(self):
        X, y = DP.arff_data('iris.arff')
        iris = datasets.load_iris()
        iris_X = iris.data
        iris_y = iris.target
        assert_array_equal(X, iris_X)
        assert(len(iris_y) == len(y))


if __name__ == '__main__':
    unittest.main()
