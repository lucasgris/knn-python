class NotFittedError(Exception):
    def __init__(self, classifier_name=None):
        if classifier_name is None:
            super().__init__(self, "Must fit {} before querying".
                             format(classifier_name))
        else:
            super().__init__(self, "Must fit the classifier before querying")
