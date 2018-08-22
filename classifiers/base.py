from abc import abstractmethod


class ClassifierBase:
    """Base class for classifiers"""

    @abstractmethod
    def fit(self, X, y):
        """Must fit the data in a predictable model"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """Must predict the class labels of a set of instances"""
        raise NotImplementedError
