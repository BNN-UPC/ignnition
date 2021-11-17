from typing import Callable


class IgnnitionException(Exception):
    """Base IGNNITION for other exceptions

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs.
        message -- explanation of the error
    """

    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f'{self.message}'

    pass


class KeywordNotFoundException(IgnnitionException):
    """Exception raised for errors in datasets.

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs.
        message -- explanation of the error
    """

    def __init__(self, keyword, file, message):
        super().__init__(message)
        self.keyword = keyword
        self.file = file
        self.message = message

    def __str__(self):
        return f'Could not fin the keyword \'{self.keyword}\' in the \'{self.file}\'. {self.message}'


class DatasetException(IgnnitionException):
    """Exception raised for errors in datasets.

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs.
        message -- explanation of the error
    """

    def __init__(self, data_path, message):
        super().__init__(message)
        self.data_path = data_path
        self.message = message

    def __str__(self):
        return f'Error found in the dataset located at \'{self.data_path}\'. {self.message}'


class DatasetFormatException(DatasetException):
    """Exception raised for formatting errors in datasets

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs
        message -- explanation of the exception
    """

    def __init__(self, data_path, message):
        super().__init__(data_path, message)
        self.data_path = data_path
        self.message = message


class DatasetNodeException(DatasetFormatException):
    """Exception raised when the path of the dataset is not found

        Attributes:
            dataset -- dataset (training, validation or predict) where the exception occurs
            path -- path of the dataset that raises the exception
            message -- explanation of the exception
        """

    def __init__(self, node_name, data_path, message):
        super().__init__(data_path, message)
        self.message = message
        self.node_name = node_name
        self.data_path = data_path

    def __str__(self):
        return f'Error found in the node {self.node_name} located at {self.data_path}. {self.message}'


class DatasetNotFoundException(DatasetException):
    """Exception raised when the path of the dataset is not found

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs
        path -- path of the dataset that raises the exception
        message -- explanation of the exception
    """

    def __init__(self, dataset, path, message):
        super().__init__(message, dataset)
        self.dataset = dataset
        self.message = message
        self.path = path

    def __str__(self):
        return f'Error found in the {self.dataset} dataset located at {self.path}. {self.message}'


class ModelDescriptionException(IgnnitionException):
    """Exception raised for errors in the model_description.yaml file.

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs.
        message -- explanation of the error
    """

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class FeatureException(IgnnitionException):
    """Exception raised for errors in the model_description.yaml file.

        Attributes:
            feature -- feature name that raises the error
            message -- explanation of the error
    """

    def __init__(self, feature, message):
        super().__init__(message)
        self.feature = feature
        self.message = message

    def __str__(self):
        return f'Error with feature \'{self.feature}\'. {self.message}'


class KeywordException(IgnnitionException):
    """Exception raised for errors in the model_description.yaml file.

        Attributes:
            feature -- feature name that raises the error
            message -- explanation of the error
    """

    def __init__(self, keyword, message):
        super().__init__(message)
        self.keyword = keyword
        self.message = message

    def __str__(self):
        return f'Error with the keyword \'{self.keyword}\'. {self.message}'


class EntityError(IgnnitionException):
    """Exception raised for errors in the model_description.yaml file.

        Attributes:
            feature -- feature name that raises the error
            message -- explanation of the error
    """

    def __init__(self, entity, entity_type, message):
        super().__init__(message)
        self.entity = entity
        self.entity_type = entity_type
        self.message = message

    def __str__(self):
        return f'Error with the entity \'{self.entity}\' used as {self.entity_type}. {self.message}'


class YAMLFormatError(IgnnitionException):
    """Exception raised for errors in the model_description.yaml file.

        Attributes:
            feature -- feature name that raises the error
            message -- explanation of the error
    """

    def __init__(self, file, file_path, message):
        super().__init__(message)
        self.file = file
        self.file_path = file_path
        self.message = message

    def __str__(self):
        return f'Error found in {self.file} at {self.file_path}. {self.message}'


class YAMLNotFoundError(IgnnitionException):
    """Exception raised for errors in the model_description.yaml file.

        Attributes:
            feature -- feature name that raises the error
            message -- explanation of the error
    """

    def __init__(self, file, file_path, message=None):
        super().__init__(message)
        self.file = file
        self.file_path = file_path
        self.message = message

    def __str__(self):
        return f'The {self.file} file was not found at {self.file_path}.'


class KerasError(IgnnitionException):
    """Exception raised for errors in the model_description.yaml file.

        Attributes:
            feature -- feature name that raises the error
            message -- explanation of the error
    """

    def __init__(self, parameter, variable, message=None):
        super().__init__(message)
        self.variable = variable
        self.parameter = parameter
        self.message = message

    def __str__(self):
        return f'Could not convert the parameter \'{self.parameter}\' into a valid tf.keras \'{self.variable}\'. ' \
               f'{self.message} '


def handle_exception(f) -> Callable:
    """Handles any possible exception raised during the execution of the decorated function

    Args:
        f: Function to decorate

    Returns:
        The decorated function
    """

    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except IgnnitionException as e:
            print(f"Caught base exception!! \nMessage: {e}")

    return wrapper
