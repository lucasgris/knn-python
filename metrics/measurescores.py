import numpy as np


class ImpurityMeasures:
    """Class for (imp)purity operations"""

    def __init__(self, X, y):
        """
        :param X:   array-like, shape = [n_samples, n_features]
            Data (Instances)
        :param y:   array-like, shape = [n_samples]
            Class labels
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.classes = np.unique(y)
        self.size = len(X)

    def percentage(self, attr, cls_label, feature):
        """
        Compute the percentage p(yi) of a given attribute xi and a given
        feature.

        :param attr: attribute value
        :param cls_label: class label
        :param feature: attribute index to compute the percentage
        :return: float [0, 1]
        """
        attr_values = list(self.X[:, feature])
        count = 0
        total = attr_values.    count(attr)
        for i in range(self.size):
            if attr_values[i] == attr and self.y[i] == cls_label:
                count += 1
        return count / total

    def percentages(self, feature):
        """
        Compute all the percentages p(yi) of a given feature.

        :param feature: attribute index to compute the percentage
        :return: numpy array shape [attribute values, class labels]
        """
        attribute_values = np.asarray(self.X[:, feature])  # 0, 1 for instance
        percentages = []  # [[p(0, y1), p(0, y2)], [p(1, y1), p(1, y2)]]
        calculated = dict()
        for attr in attribute_values:
            partial = []
            for yi in self.classes:
                if (attr, yi) not in calculated.keys():
                    calculated[(attr, yi)] = self.percentage(attr, yi, feature)
                partial.append(calculated[(attr, yi)])
            percentages.append(partial)
        return np.asarray(percentages)

    @staticmethod
    def _log2(x):
        # Compute the log. If the result is indeterminate, return 0.
        if x == 0:
            return 0
        return np.log2(x)

    def gini(self, feature):
        """
        Compute the gini measures of a given feature in the data set.

        :param feature: feature to compute
        :return: numpy array shape [attribute values]
        """
        percentages = self.percentages(feature)
        ginis = []
        for per in percentages:
            ginis.append(1 - np.sum(list(map(lambda p: p**2, per))))
        return np.asarray(ginis)

    def entropy(self, feature):
        """
        Compute the entropy measures of a given feature in the data set.

        :param feature: feature to compute
        :return: numpy array shape [attribute values]
        """
        percentages = self.percentages(feature)
        entropies = []
        for per in percentages:
            entropies.\
                append((-1) * np.sum(list(map(lambda p: p * ImpurityMeasures.
                                              _log2(p), per))))
        return np.asarray(entropies)

    def classification_error(self, feature):
        """
        Compute the classification error measures of a given feature in the
        data set.

        :param feature: feature to compute
        :return: numpy array shape [attribute values]
        """
        percentages = self.percentages(feature)
        errors = []
        for per in percentages:
            errors.append(1 - per.max())
        return np.asarray(errors)
