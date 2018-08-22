import unittest
from metrics.measurescores import *


class Tests(unittest.TestCase):
    # For 2 classes test

    X = np.asarray([[1, 1, 0],
                    [1, 0, 1],
                    [1, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1]])
    y = ['class1', 'class2', 'class2', 'class1', 'class2']
    im = ImpurityMeasures(X, y)

    def test_percentages(self):
        per_feature = self.im.percentages(0)
        for p in per_feature:
            assert(np.sum(p) == 1)

    def test_gini(self):
        ginis_feature = self.im.gini(0)
        for g in ginis_feature:
            assert(g <= 0.5)
            assert(g >= 0)

    def test_entropy(self):
        entropies_feature = self.im.entropy(0)
        for e in entropies_feature:
            assert (e <= 1)
            assert (e >= 0)

    def test_error_class(self):
        errors_feature = self.im.classification_error(0)
        for e in errors_feature:
            assert (e <= 0.5)
            assert (e >= 0)


if __name__ == '__main__':
    unittest.main()
