"""
The :mod:'neighbors' module implements the k-nearest neighbors algorithm.
"""
import numpy as np
import warnings

from math import sqrt
from abc import abstractmethod

from classifiers.base import ClassifierBase
from classifiers.exceptions import NotFittedError


class DistanceMetric:
    """
    DistanceMetric class

    This class provides a uniform interface to fast distance metric functions.
    The various metrics can be accessed via the get_metric class method and the
    metric string identifier (see below).

    Available Metrics

    The following lists the string metric identifiers and the associated
    distance metric classes:

    Metrics intended for real-valued vector spaces:

    identifier	    class name	            distance function
    "euclidean"	    EuclideanDistance	    sqrt(sum((x - y)^2))
    "manhattan"	    ManhattanDistance	    sum(|x - y|)
    "chebyshev"	    ChebyshevDistance	    max(|x - y|)
    """

    @staticmethod
    def get_metric(metric):
        """
        Get the given distance metric from the string identifier.

        See the docstring of DistanceMetric for a list of available metrics.

        The distance metric to use
        :param      metric: string or class name
        :return:    A DistanceMetric object
        """
        if metric == 'euclidean':
            return DistanceMetric.EuclideanDistance()
        elif metric == 'manhattan':
            return DistanceMetric.ManhattanDistance()
        elif metric == 'chebyshev':
            return DistanceMetric.ChebyshevDistance()
        else:
            raise ValueError("Not valid metric identifier")

    class _Distance:
        """Abstract class for Distance class implementations."""
        @staticmethod
        def _dist_vectors(x, y):
            if not type(x) is np.ndarray:
                x = np.array(x)
            if not type(y) is np.ndarray:
                y = np.array(y)
            return x, y

        @abstractmethod
        def __call__(self, x, y):
            """
            Compute the distances between x and y.

            :param x: Array of shape (Nx, D), representing Nx points in
                D dimensions.
            :param y: Array of shape (Nx, D), representing Nx points in
                D dimensions.
            :return: The computed distance.
            """
            raise NotImplementedError

        def pairwise(self, X):
            """
            Compute the pairwise distances between X.

            :param X: Array of shape (Nx, D), representing Nx points in
                D dimensions.
            :return: dist : ndarray
                The shape (Nx) array of pairwise distances between points in X
            """
            dists = np.ndarray([len(X), len(X)])
            for i in range(0, len(X)):
                for j in range(0, len(X)):
                    dists[i][j] = self(X[i], X[j])
            return dists

    # Implementations of Distances:
    class EuclideanDistance(_Distance):
        def __call__(self, x, y):
            x, y = super()._dist_vectors(x, y)
            return sqrt(np.sum((x - y)**2))

        def pairwise(self, X):
            return super().pairwise(X)

    class ManhattanDistance(_Distance):
        def __call__(self, x, y):
            x, y = super()._dist_vectors(x, y)
            return np.sum(np.abs(x - y))

        def pairwise(self, X):
            return super().pairwise(X)

    class ChebyshevDistance(_Distance):
        def __call__(self, x, y):
            x, y = super()._dist_vectors(x, y)
            return np.max(np.abs(x - y))

        def pairwise(self, X):
            return super().pairwise(X)


def brute_force_k_neighbors(data, x, ids=None, n_neighbors=5,
                            return_distance=True, dist_f=None, ignore_x=False):
    """
    Perform a brute force search for the k nearest neighbors.

    :param data: a data to perform the operation
        Note: all features of data will be considered as a dimension
        to calculate the distances

    :param x: array-like
        A point to query

    :param ids: array-like
        ID's array of each element of data. If none, the indexes of the data
        array will be considered and returned as a result of the query.

    :param n_neighbors: integer (default = 1)
        The number of nearest neighbors to return

    :param return_distance: boolean (default = True)
        If True, return a tuple (d, i) of distances and indices if False,
        return array i

    :param dist_f: callable (default=None)
        A function distance to calculate the distance between points.
        If no function is provided, the euclidean distance will be used.

    :param ignore_x: boolean, optional (default=False)
            If True, ignore the instances in X when computing the distances.
            This is useful when the fit data set contains X.

    :return:
        i       :   if return_distance == False
        (d,i)   :   if return_distance == True
        d       :   array of doubles - shape: X.shape[:-1] + (k,)
                    each entry gives the list of distances to the
                    neighbors of the corresponding point
        i       :   array of integers - shape: X.shape[:-1] + (k,)
                    each entry gives the list of ids/indices of neighbors
                    of the corresponding point
    """

    # Every query that calculate the k nearest neighbors in this module ends
    # in this function.
    #
    # If the result of the query consists in one array of distances and one
    # array of indices, each index in the two arrays will represent the same
    # instance of the data set.
    #
    # To predict the class of some instance, identify the classes of the
    # nearest neighbors.
    #
    # TODO: OPTIMIZE THIS FUNCTION
    warnings.warn("deprecated", DeprecationWarning)

    if ignore_x:
        warnings.warn('There is a bug in ignore x feature and the code has been '
                      'commented. The ignore_x parameter has no effect in this '
                      'function')

    if dist_f is None:  # If distance function is not set
        dist_f = DistanceMetric.get_metric('euclidean')

    closest_neighbors = []  # Array of closest neighbors
    if ids is not None:
        for i in range(len(data)):
            # Apply the distance function to each point and save the results:
            closest_neighbors.append([ids[i], dist_f(x, data[i])])

    else:  # Consider the id the index of each instance in data:
        for i in range(len(data)):
            # Apply the distance function to each point and save the results:
            closest_neighbors.append([i, dist_f(x, data[i])])

    # Sort based on distances
    closest_neighbors.sort(key=lambda n: n[1])  # Distances: second column

    if ignore_x:  # Bug: If another point have the same value as x, the wrong
                  # instance might be deleted. The predicted result will not 
                  # change, but the id of the neighbor will not be correct.
        # del(closest_neighbors[0])
        pass

    closest_neighbors = np.asarray(closest_neighbors)

    ids = closest_neighbors[:, 0]           # IDs: first column
    distances = closest_neighbors[:, 1]     # Distances: second column

    if return_distance:
        return distances[:n_neighbors], \
               np.ndarray.astype(ids[:n_neighbors], dtype='int64')
    else:
        return ids[:n_neighbors]


class KDTree:
    """
    KDTree for fast generalized N-point problems.
    """

    class Node:
        """KD-tree node data structure"""

        def __init__(self, node_parent=None, data=None, ids=None, depth=0):
            """
            :param node_parent: the parent node of this node (default=None).
            :param data: the data of the node (default=None).
            :param ids: an array-like of ID's (default=None).
            """
            self.parent = node_parent
            self._data = data
            if data is not None:
                self._count = len(data)
            self._condition = None
            self._right = None
            self._left = None
            if ids is not None:
                self._ids = ids
            else:
                self._ids = list(range(len(data)))
            self._depth = depth
            self._index = 0

        @property
        def left(self):
            """Return the left child node of this node"""
            return self._left

        @property
        def right(self):
            """Return the right child node of this node"""
            return self._right

        @property
        def count(self):
            """Return the amount of elements of this node"""
            return self._count

        @property
        def data(self):
            """Return the data of this node"""
            return self._data

        @property
        def ids(self):
            """Return the indexes array of the data"""
            return self._ids

        @property
        def condition(self):
            """
            Return the condition of this node that represents a condition which
            an arbitrary subtree must satisfy.

            The condition is a callable object that must evaluate if a provided
            value should belong to an arbitrary subtree (if True), or the other
            subtree (if False).

            Use this method to walk over the tree, going to the right or left
            side of it when the condition is satisfied.
            """
            return self._condition

        @condition.setter
        def condition(self, value):
            if not callable(value):
                raise ValueError("Not a callable object")
            self._condition = value

        @data.setter
        def data(self, value):
            """Sets the data of this node"""
            self._data = np.asarray(value)
            self._count = len(self.data)

        @right.setter
        def right(self, value):
            """Sets the right child node"""
            self._right = value

        @left.setter
        def left(self, value):
            """Sets the left child node"""
            self._left = value

        @ids.setter
        def ids(self, value):
            """Sets the indexes array of the data"""
            self._ids = value

        def __getitem__(self, instance_id):
            return self._data[self._ids.index(instance_id)]

        def __len__(self):
            return len(self._data)

        def __str__(self):
            return "KDTree-node [depth={}, \nid={}]".format(
                self._depth, self._ids)

    class KDTreeBuilder:
        """A KD-Tree builder class"""

        def __init__(self, raw_data, leaf_size):
            """
            :param raw_data: an array of elements which will compose the tree.
            :param leaf_size: the maximum size of the leafs.
            """
            self.data = raw_data
            self.leaf_size = leaf_size
            self.features = len(self.data[0])

        def build(self, node, split_feature=0, depth=0):
            """Build nodes recursively"""
            # TODO: OPTIMIZE THIS CODE

            current_feature = split_feature
            if node.count > self.leaf_size:
                # Find median:
                feature = current_feature
                median = np.sum(np.asarray(node.data)[:, feature]) / node.count

                # Filter elements of each subtree:
                rdt = []  # Array of data (right subtree)
                ldt = []  # Array of data (left subtree)
                r_ids = []  # Array of IDs (right subtree)
                l_ids = []  # Array of IDs (left subtree)
                for i in range(len(node)):
                    if node.data[i, feature] >= median:
                        rdt.append(node.data[i])
                        r_ids.append(node.ids[i])
                    else:
                        ldt.append(node.data[i])
                        l_ids.append(node.ids[i])

                # Assign values:
                node.right = KDTree.Node(node_parent=node, data=np.asarray(rdt),
                                         depth=depth + 1, ids=np.asarray(r_ids))
                node.left = KDTree.Node(node_parent=node, data=np.asarray(ldt),
                                        depth=depth + 1, ids=np.asarray(l_ids))
                # Set condition of parent node:
                node.condition = lambda val: val[feature] >= median
                # Make subtrees recursively:
                if current_feature == self.features - 1:
                    current_feature = 0
                else:
                    current_feature += 1

                self.build(node.right, split_feature=current_feature,
                           depth=depth + 1)
                self.build(node.left, split_feature=current_feature,
                           depth=depth + 1)

        def __call__(self):
            self.root = KDTree.Node(node_parent=None, data=self.data)
            self.build(self.root)
            return self.root

    def __init__(self, X, leaf_size=40, metric='euclidean'):
        """
        :param X:   array-like, shape = [n_samples, n_features]

            n_samples is the number of points in the data set,
            and n_features is the dimension of the parameter space.

        :param leaf_size: Number of points at which to switch to brute-force.

        :param metric: string or callable

            The distance metric to use for the tree.

            See the documentation of the DistanceMetric class for a list
            of available metrics.
            Default='euclidean'
        """
        self._X = X
        self._dim = len(X[0])
        self._leaf_size = leaf_size
        if callable(metric):
            self._metric = metric
        else:
            self._metric = DistanceMetric.get_metric(metric)

        # Builds the KD-Tree:
        builder = self.KDTreeBuilder(self._X, self._leaf_size)
        self.tree = builder()

    @staticmethod
    def _breadth_first(node_root, x):
        # Return the leaf that contains nearest neighbors, performing a breadth
        # first search
        # x: search key
        #
        # NOTE: attempt to implement breadth first search here, but of course
        # this is not a good implementation of it.
        #
        # WARNING: THIS METHOD HAS NOT BEEN TESTED
        # TODO: TEST AND FIX IF NECESSARY

        nodes = list()
        nodes.append(node_root)
        # While exists parents in nodes:
        while len(list(filter(lambda n: n.left is None and n.right is None,
                              nodes))) != len(nodes):
            parent = nodes.pop(0)
            if parent.left is not None:
                nodes.append(parent.left)
            if parent.right is not None:
                nodes.append(parent.right)
            # Filter only nodes that satisfies the condition:
            filter(lambda n: n.condition(x), nodes)

        # Should have only one node:
        return nodes.pop(0)

    @staticmethod
    def _depth_first(node_root, x):
        # Return the leaf that contains nearest neighbors, performing a depth
        # first search
        # x: search key

        node = node_root

        # Find leafs:
        while node.left is not None and node.right is not None:
            # While node is not a leaf:
            if node.condition(x):
                node = node.right
            else:
                node = node.left
        # The node is a leaf, and satisfies all the conditions

        return node

    def _query_point(self, x, k=1, return_distance=True, breadth_first=False,
                     ignore_x=False):
        # Evaluate the nearest neighbor of a point:

        # TODO: CHECK IF SEARCH METHODS ARE CORRECT
        if breadth_first:
            node = KDTree._breadth_first(self.tree, x)
        else:
            node = KDTree._depth_first(self.tree, x)

        while node.count < k:
            node = node.parent

        # Perform a brute search for closest neighbors:
        if return_distance:
            dists_near, idx_near = \
                brute_force_k_neighbors(node.data, x, node.ids,
                                        k, return_distance, self._metric,
                                        ignore_x)
            return np.asarray(dists_near), np.asarray(idx_near)
        else:
            idx_near = \
                brute_force_k_neighbors(node.data, x, node.ids,
                                        k, return_distance, self._metric,
                                        ignore_x)
            return np.asarray(idx_near)

    def query(self, X, k=1, return_distance=True, breadth_first=False,
              ignore_x=False):
        """
        :param X: array-like, last dimension self.dim
            An array of points to query

        :param k: integer (default = 1)
            The number of nearest neighbors to return

        :param return_distance: boolean (default = True)
            If True, return a tuple (d, i) of distances and indices if False,
            return array i

        :param breadth_first:  boolean (default = False)
            If True, then query the nodes in a breadth-first manner.
            Otherwise, query the nodes in a depth-first manner.

        :param ignore_x: boolean, optional (default = False)
            If True, ignore the instances in X when computing the distances.
            This is useful when the fit data set contains X.

        :return:
            i       :   if return_distance == False
            (d,i)   :   if return_distance == True
            d       :   array of doubles - shape: X.shape[:-1] + (k,)
                        each entry gives the list of distances to the
                        neighbors of the corresponding point
            i       :   array of integers - shape: X.shape[:-1] + (k,)
                        each entry gives the list of indices of neighbors
                        of the corresponding point
        """
        idx_near = []
        dists_near = []
        for x in X:
            # Compute the closest neighbors and append to the respective
            # query results:
            if return_distance:
                dists, idx = self._query_point(x, k, return_distance,
                                               breadth_first, ignore_x)
                idx_near.append(idx)
                dists_near.append(dists)
            else:
                idx = self._query_point(x, k, return_distance, breadth_first,
                                        ignore_x)
                idx_near.append(idx)

        if return_distance:
            return np.asarray(dists_near), np.asarray(idx_near)
        else:
            return np.asarray(idx_near)


class KNeighborsClassifier(ClassifierBase):
    """
    Unsupervised learner for implementing neighbor searches.
    """
    def __init__(self, n_neighbors=5, algorithm='kd_tree', leaf_size=30,
                 weights='uniform', dist_metric='euclidean'):
        """
        :param n_neighbors:  int, optional (default = 5)
            Number of neighbors to use by default for k_neighbors queries.

        :param algorithm: {'kd_tree', 'brute'}, optional (default='kd_tree')
            Algorithm used to compute the nearest neighbors:
                'kd_tree' will use KDTree
                'brute' will use a brute-force search.

        :param leaf_size: int, optional (default = 30)
            Leaf size passed to BallTree or KDTree. This can affect the speed
            of the construction and query, as well as the memory required to
            store the tree. The optimal value depends on the nature of the
            problem.

        :param weights: weight function used in prediction. Possible values:
            'uniform' : uniform weights. All points in each neighborhood are
            weighted equally.
            'distance' : weight points by the inverse of their distance,
            in this case, closer neighbors .
            callable : a callable object

        :param dist_metric: string or callable, (default 'euclidean') metric to
            use for distance computation.
        """
        self._n_neighbors = n_neighbors
        if algorithm == 'kd_tree':
            self._algorithm = KDTree
            self._tree = None
        elif algorithm == 'brute':
            self._algorithm = None
        else:
            raise ValueError("Not valid algorithm")

        if weights == 'uniform':
            self._weight_function = None
        elif weights == 'distance':
            # If two instances in the data set have the same point values,
            # divide only by their distance will generate a division by zero
            # error.
            self._weight_function = lambda dist: 1. / (dist + 0.0000000001)
        elif not callable(weights):
            raise ValueError("Not valid weights argument")
        else:
            self._weight_function = weights

        self._dist_metric = dist_metric
        self._classes = []  # Relates ids and classes
        self._data = []     # Relates ids and instances
        self._leaf_size = leaf_size

    @property
    def ids(self):
        """
        Return the ids of the data set
        """
        return range(0, len(self._data))

    @property
    def data_classes(self):
        """
        Return the classes of the data set of each instance
        """
        return self._classes

    @property
    def data_instances(self):
        """
        Return the instance with the feature values
        """
        return self._data

    @property
    def classes(self):
        """
        Return the class labels of the data set
        """
        return np.unique(self._classes)

    def fit(self, X, y):
        """
        Fit the model using X as training data.
        :param X: {array-like, KDTree}
            Training data.
        :param y: array-like
            Classes of the respective training data.
        """
        if len(X) != len(y):
            raise ValueError("y must contain the exactly equal number of "
                             "classes of each instance in X")
        # Store the reference of each instance in each class array
        # Note: this might not be a good method to fit the model
        self._data = np.asarray(X)
        self._classes = np.asarray(y)

        # If the set algorithm is the kd-tree, build the tree
        if self._algorithm == KDTree:
            self._tree = KDTree(self._data, self._leaf_size, self._dist_metric)

        # If the set algorithm is a brute force, pass:
        # The k nearest neighbors algorithm is a lazy algorithm,
        # that is, is not necessary to train the data to perform
        # predictions.

    def k_neighbors(self, X, n_neighbors, return_distance=True, ignore_x=False):
        """
        Finds the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        :param X: array-like, shape (n_query, n_features)
            The query point or points. If not provided, neighbors of each
            indexed point are returned. In this case, the query point is
            not considered its own neighbor.

        :param n_neighbors: int
            Number of neighbors to get (default is the value passed to the
            constructor).

        :param return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        :param ignore_x: boolean, optional. Defaults to False.
            If True, ignore the instances in X when computing the distances.
            This is useful when the fit data set contains X.

        :return:
            dist : array
            Array representing the lengths to points, only present if
            return_distance=True
            ind : array
            Indices of the nearest points in the population matrix.
        """
        if self._algorithm == KDTree:  # Query the results using the KDTree:
            return self._tree.query(X=X, k=n_neighbors,
                                    return_distance=return_distance,
                                    ignore_x=ignore_x)
        else:  # Query the results performing a brute force search
            idx_near = []
            dists_near = []
            if return_distance:
                for x in X:
                    # Compute the closest neighbors and append to the respective
                    # query results:
                    d, i = brute_force_k_neighbors(data=self.data_instances, x=x,
                                                   ids=None, n_neighbors=n_neighbors,
                                                   return_distance=return_distance,
                                                   ignore_x=ignore_x)
                    idx_near.append(i)
                    dists_near.append(d)
                return np.asarray(dists_near), \
                       np.asarray(idx_near, dtype='int64')
            
            else:
                for x in X:
                    # Compute the closest neighbors and append to the respective
                    # query results:
                    i = brute_force_k_neighbors(data=self.data_instances, x=x,
                                                ids=None, n_neighbors=n_neighbors,
                                                return_distance=return_distance,
                                                ignore_x=ignore_x)
                    idx_near.append(i) 
                
                return np.asarray(idx_near, dtype='int64')

    def predict(self, X, ignore_x=False):
        """
        Predict the class labels for the provided data.

        :param X: array-like, shape (n_query, n_features)
            Test samples.

        :param ignore_x: boolean, optional. Defaults to False.
            If True, ignore the instances in X when computing the distances.
            This is useful when the fit data set contains X.

        :return: array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        if self._data is None:
            raise NotFittedError('neighbors')

        # Calculate the distance between each instance in X and compute
        # the class label based on the closest n neighbors of the fit model
        dists_near, idx_near = self.k_neighbors(X=X, ignore_x=ignore_x,
                                                n_neighbors=self._n_neighbors,
                                                return_distance=True)

        y_pred = []         # Relates each voting result (a class prediction)
        # with each instance in X

        # If weights function is defined, apply it for each obtained distance:
        if callable(self._weight_function):
            weights = np.array(list(map(self._weight_function, dists_near)))
            # Vote for majority class and make predictions:
            for query in range(len(idx_near)):
                voting = dict()  # Relates each class and the amount of nearest
                # neighbors of it
                for cls in self.data_classes:
                    voting[cls] = 0  # Reset pooling
                for i in range(len(idx_near[query])):
                    voting[self.data_classes[idx_near[query][i]]] = \
                        weights[query][i]
                y_pred.append(max(voting, key=voting.get))
        else:
            # Vote for majority class and make predictions:
            for query in range(len(idx_near)):
                voting = dict()  # Relates each class and the amount of nearest
                # neighbors of it
                for cls in self.data_classes:
                    voting[cls] = 0  # Reset pooling
                for index in idx_near[query]:
                    voting[self.data_classes[index]] += 1
                y_pred.append(max(voting, key=voting.get))
        return np.asarray(y_pred)
