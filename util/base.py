from abc import abstractmethod


class DataParserBase:
    """Data parser. Must parse data from files."""
    @staticmethod
    @abstractmethod
    def parse(file_path):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arff_data(file_path):
        raise NotImplementedError
