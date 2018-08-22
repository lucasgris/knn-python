import argparse
import os

from classifiers.neighbors import KNeighborsClassifier
from util.dataparser import DataParser


def main(args):

    algorithm = 'brute'
    weights = 'uniform'
    if args.knn_kdt:
        algorithm = 'kd-tree'
    elif args.knn_brute_w:
        algorithm = 'brute'
        weights = 'distance'
    elif args.knn_kdt_w:
        weights = 'distance'

    if args.d == 0:
        dist_metric = 'chebyshev'
    elif args.d == 1:
        dist_metric = 'manhattan'
    else:
        dist_metric = 'euclidean'

    classifier = KNeighborsClassifier(n_neighbors=args.k, algorithm=algorithm,
                                      weights=weights, dist_metric=dist_metric)

    dp = DataParser
    if args.data_parser_module is not None:
        import importlib
        data_parser = importlib.import_module(args.data_parser_module)
        dp = data_parser.DataParser

    ids_train, X_train, y_train = dp.parse(args.train_data_path)
    classifier.fit(X_train, y_train)

    if args.test_data_path is not None:
        ids_test, X_test, t_test = dp.parse(args.test_data_path)
    else:
        ids_test, X_test, t_test = ids_train, X_train, y_train

    # # ignore_x bug found
    # y_pred = classifier.predict(X_test, ignore_x=True)
    # closest_neighbors = classifier.k_neighbors(X_test, args.k, 
                                                # return_distance=False)
    # for i in range(len(y_pred)):
    #   print('class: {}\nclosest neighbors: {}'.format(y_pred, 
    #          closest_neighbors))
    
    # # TODO: Remove the code below after ignore_x implementation
    X_train = list(X_train)
    y_train = list(y_train)
    closest_neighbors = []
    for i in range(len(X_train)):
        x = X_train.pop(i)
        y = y_train.pop(i)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict([x])[0]
        closest_neighbors = classifier.k_neighbors([x], 
                            args.k, return_distance=False)[0]
        X_train.insert(i, x)
        y_train.insert(i, y)
        closest_neighbors[i:] += 1
        print('classe: {}\nvizinhos mais próximos: {}'.format(y_pred, 
              closest_neighbors))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Classification algorithms.')

    # Optional arguments:
    parser.add_argument('--knn', help='executes the k nearest neighbor '
                                      'classification algorithm with default '
                                      'hyper parameters (brute force, '
                                      'uniform distance)',
                        action='store_true', default=True)
    parser.add_argument('--knn_kdt', help='executes the k nearest neighbor '
                                          'classification using the kd-tree '
                                          'algorithm.',
                        action='store_true')
    parser.add_argument('--knn_kdt_w', help='executes the k nearest neighbor '
                                            'classification algorithm '
                                            'performing a kd tree search and '
                                            'considering the weighted distance '
                                            'between instances',
                        action='store_true')
    parser.add_argument('--knn_brute_w', help='executes the k nearest neighbor '
                                              'classification algorithm '
                                              'performing a kd tree search and '
                                              'considering the weighted '
                                              'distance between instances',
                        action='store_true')

    parser.add_argument('-k', help='sets the k parameter', type=int, default=5)
    parser.add_argument('-d', help='sets the distance measure to use in the '
                                   'algorithm, that is, 0 - supreme, '
                                   '1 - manhattan, 2 - euclidean',
                        type=int, choices=[0, 1, 2], default=2)
    parser.add_argument('train_data_path', help='The path of the input data to '
                                                'train')
    parser.add_argument('--test_data_path', help='The path of the input data '
                                                 'to test. If not set, '
                                                 'the train data will be used')
    parser.add_argument('--data_parser_module', help='The path of a data '
                                                     'parser python module')

    if not os.path.isfile(parser.parse_args().train_data_path):
        print('Not able to get the data')
        exit(1)

    main(parser.parse_args())
