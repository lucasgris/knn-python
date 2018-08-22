import argparse
import os
import importlib
import warnings

from metrics.measurescores import ImpurityMeasures
from util.dataparser import DataParser


def main(args):

    dp = DataParser
    if args.data_parser_module is not None:
        data_parser = importlib.import_module(args.data_parser_module)
        dp = data_parser.DataParser

    ids, X, y = dp.parse(args.data_path)

    pm = ImpurityMeasures(X, y)

    pandas_found = importlib.find_loader('pandas')
    if pandas_found is not None:
        pandas_lib = importlib.import_module('pandas')
        pd = pandas_lib.pandas
        d = dict()
        d['Attribute values'] = X[:, args.a]
        if args.imp == 1 or args.imp is None:
            d['Ginis of attribute {}'.format(args.a)] = pm.gini(args.a)
        if args.imp == 2 or args.imp is None:
            d['Entropies of attribute {}'.format(args.a)] = pm.entropy(args.a)
        if args.imp == 3 or args.imp is None:
            d['Classification errors of attribute {}'.format(args.a)] = \
            pm.classification_error(args.a)
        d['Instance class'] = y
        df = pd.DataFrame(data=d)
        print(df)
    else:
        warnings.warn('For better visualization, install "pandas" module.')
        print("Showing results for each atribute value:")
        print("Values for atribute {}:{}".format(args.a, X[:,args.a]))
        if args.imp is None:
            print("gini(attribute {})={}"
                "\nentropy(attribute {})={}"
                "\nclassification error(attribute {})={}".
                format(args.a, pm.gini(args.a), args.a, pm.entropy(args.a), args.a,
                        pm.classification_error(args.a)))
        elif args.imp == 1:
            print("gini(attribute {})={}".format(args.a, pm.gini(args.a)))
        elif args.imp == 2:
            print("entropy(attribute {})={}".format(args.a, pm.entropy(args.a)))
        elif args.imp == 3:
            print("classification error(attribute {})={}".format(args.a, 
                pm.classification_error(args.a)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metric measures scores of '
                                                 'data algorithms.')

    # Optional arguments:
    parser.add_argument('--imp', help='The impurity measure, 1 - gini, '
                                      '2 - entropy, 3 - classification error',
                        type=int, choices=[1, 2, 3], default=None)
    parser.add_argument('-a', help='sets the feature that will be measured',
                        type=int)
    parser.add_argument('data_path', help='The path of the input data')
    parser.add_argument('--data_parser_module', help='The path of a data '
                                                     'parser python module')

    if not os.path.isfile(parser.parse_args().data_path):
        print('Not able to get the data')
        exit(1)

    main(parser.parse_args())
