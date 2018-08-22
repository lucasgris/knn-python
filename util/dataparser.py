# TODO: arff parsing
"""Default data parser module"""
import numpy as np

from util.base import DataParserBase


class DataParser(DataParserBase):
    """Data parser class. Parses data from files."""
    @staticmethod
    def parse(file_path, id_column=None, class_column=None, auto=True):
        """
        Default data parser. Parses a file into features and classes instances.

        Example of input file:

        id sepallength sepalwidth petallength petalwidth class
        0          5.1        3.5         1.4        0.2 setosa
        1          7.0        3.2         4.7        1.4 versicolor
        2          6.3        3.3         6.0        2.5 virginica


        :param file_path            The file path to parse the data.

        :param id_column            The column with ids of each instance.

        :param class_column         The column with classes of each instance.

        :param auto                 The first column will be considered as
                                    the id column, and the last will be
                                    considered as the class column.
        :return:
            ids, X, classes         An array with each id, an matrix with each
                                    instance containing its own values of
                                    features, and an array with each class.
                                    If the id_column or class_column is not
                                    being provided while auto is False,
                                    None will be returned.
        """
        n_columns = None
        X = []
        ids = None
        class_labels = None

        with open(file_path) as file:
            # get the number of columns:
            for line in file:
                data = line.split()
                n_columns = len(data)
                break

            if auto:
                id_column = 0
                class_column = n_columns - 1

            if id_column is not None:
                ids = []
            else:
                id_column = -1

            if class_column is not None:
                class_labels = []
            else:
                class_column = n_columns

            for line in file:
                data = line.split()
                X.append(data[id_column + 1:class_column])
                if ids is not None:
                    ids.append(data[id_column])
                if class_labels is not None:
                    class_labels.append(data[class_column])

        return np.asarray(ids).astype(int), \
               np.asarray(X).astype(float), \
               np.asarray(class_labels)

    @staticmethod
    def arff_data(file_path, attr_type='Float64'):
        """
        Parses a weka file.

        :param file_path: a path to an .arff file
        :param attr_type: attribute type of the attribute values.
            Default to Float64. If None, no casting will be made.

        :type attr_type: Any

        :return: X, class labels
        """
        X = []
        class_labels = []
        with open(file_path) as file:
            is_data = False
            lines = file.readlines()
            for line in lines:
                if len(line) >= 1 and line[0] == '%':
                    continue
                elif "@DATA" in line.upper():
                    is_data = True
                elif is_data:
                    raw = line.split(',')
                    if attr_type is not None:
                        X.append(np.asarray(raw[:len(raw) - 1]).
                                 astype(dtype=attr_type))
                    else:
                        X.append(np.asarray(raw[:len(raw) - 1]))
                    class_labels.append(raw[len(raw) - 1][:-1])
        return np.asarray(X), np.asarray(class_labels)
