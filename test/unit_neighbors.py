import unittest
from numpy.testing import assert_array_equal
import numpy as np
import sklearn.neighbors
import classifiers.neighbors
from sklearn.neighbors import DistanceMetric as sklearn_dm
from classifiers.neighbors import DistanceMetric as dm


class TestNeighbors(unittest.TestCase):

    def test_euclidean(self):
        X = [[0, 1, 2], [3, 4, 5]]
        assert_array_equal(sklearn_dm.get_metric('euclidean').pairwise(X),
                           dm.get_metric('euclidean').pairwise(X))

    def test_manhattan(self):
        X = [[0, 1, 2], [3, 4, 5]]
        assert_array_equal(sklearn_dm.get_metric('manhattan').pairwise(X),
                           dm.get_metric('manhattan').pairwise(X))

    def test_chebyshev(self):
        X = [[0, 1, 2], [3, 4, 5]]
        assert_array_equal(sklearn_dm.get_metric('chebyshev').pairwise(X),
                           dm.get_metric('chebyshev').pairwise(X))

    def test_brute_1(self):
        samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]

        # From sk learn:
        neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
        neigh.fit(samples)
        dists_skl, ids_skl = neigh.kneighbors([[1., 1., 1.]])
        # From neighbors.classifiers
        dists, ids = classifiers.neighbors.brute_force_k_neighbors(samples, x=[1., 1., 1.], n_neighbors=1)
        # The brute_force_k_neighbors function only return results
        # for one query, that is the reason why the results from sk learn
        # are unpacked below:
        assert_array_equal(ids, ids_skl[0])
        assert_array_equal(dists, dists_skl[0])

    def test_brute_2(self):
        samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]

        # From sk learn:
        neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=2)
        neigh.fit(samples)
        dists_skl, ids_skl = neigh.kneighbors([[1., 1., 1.]])
        # From neighbors.classifiers
        dists, ids = classifiers.neighbors.brute_force_k_neighbors(samples, x=[1., 1., 1.], n_neighbors=2)
        # The brute_force_k_neighbors function only return results
        # for one query, that is the reason why the results from sk learn
        # are unpacked below:
        assert_array_equal(ids, ids_skl[0])
        assert_array_equal(dists, dists_skl[0])

    def test_brute_3(self):
        samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]

        # From sk learn:
        neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=3)
        neigh.fit(samples)
        dists_skl, ids_skl = neigh.kneighbors([[1., 1., 1.]])
        # From neighbors.classifiers
        dists, ids = classifiers.neighbors.brute_force_k_neighbors(samples, x=[1., 1., 1.], n_neighbors=3)
        # The brute_force_k_neighbors function only return results
        # for one query, that is the reason why the results from sk learn
        # are unpacked below:
        assert_array_equal(ids, ids_skl[0])
        assert_array_equal(dists, dists_skl[0])

    def test_kdtree(self):
        # WARN: THIS TEST MIGHT FAIL
        # The sk learn algorithm is more precise than this one.
        X = np.random.random((100, 3))  # 100 points in 3 dimensions
        skdt = sklearn.neighbors.KDTree(X, leaf_size=20)
        sk_dist, sk_ind = skdt.query([X[0]], k=3)
        kdt = classifiers.neighbors.KDTree(X, leaf_size=20)
        dist, ind = kdt.query([X[0]], k=3)
        assert_array_equal(sk_dist, dist)
        assert_array_equal(sk_ind, ind)


if __name__ == '__main__':
    unittest.main()
