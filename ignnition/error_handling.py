from typing import Callable
import sys


class IgnnitionException(Exception):
    """Base IGNNITION for other exceptions

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs.
        message -- explanation of the error
    """

    def __init__(self, message):
        super(IgnnitionException, self).__init__(message)
        self.message = message

    def __str__(self):
        return f'{self.message}'

    pass


class NormalizationException(IgnnitionException):
    """Exception for normalization errors

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs.
        message -- explanation of the error
    """

    def __init__(self, norm_function, message):
        super(NormalizationException, self).__init__(message)
        self.norm_function = norm_function
        self.message = message

    def __str__(self):
        return f'There was an while applying the {self.norm_function} function for normalizing the data. {self.message}'

    pass

class DenormalizationException(IgnnitionException):
    """Exception for normalization errors

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs.
        message -- explanation of the error
    """

    def __init__(self, denorm_function, message):
        super(DenormalizationException, self).__init__(message)
        self.denorm_function = denorm_function
        self.message = message

    def __str__(self):
        return f'There was an while applying the {self.denorm_function} function for denormalizing the data.' \
               f' {self.message}'

    pass

class CheckpointException(IgnnitionException):
    """Exception for normalization errors

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs.
        message -- explanation of the error
    """

    def __init__(self, message):
        super(CheckpointException, self).__init__(message)
        self.message = message

    def __str__(self):
        return f'{self.message}'

    pass

class OutptuLabelException(IgnnitionException):
    """Exception for normalization errors

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs.
        message -- explanation of the error
    """

    def __init__(self, message):
        super(OutptuLabelException, self).__init__(message)
        self.message = message

    def __str__(self):
        return f'{self.message}'

    pass

class CheckpointNotFoundException(CheckpointException):
    """Exception for normalization errors

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs.
        message -- explanation of the error
    """

    def __init__(self, path, message):
        super(CheckpointNotFoundException, self).__init__(message)
        self.path = path
        self.message = message

    def __str__(self):
        return f'Could not find a valid checkpoint file at {self.path}. {self.message}'

    pass


class CheckpointRequiredException(IgnnitionException):
    """Exception for normalization errors

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs.
        message -- explanation of the error
    """

    def __init__(self, message):
        super(CheckpointRequiredException, self).__init__(message)
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
        super(KeywordNotFoundException, self).__init__(message)
        self.keyword = keyword
        self.file = file
        self.message = message

    def __str__(self):
        return f'Could not find the keyword \'{self.keyword}\' in the \'{self.file}\'. {self.message}'


class DatasetException(IgnnitionException):
    """Exception raised for errors in datasets.

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs.
        message -- explanation of the error
    """

    def __init__(self, data_path, message):
        super(DatasetException, self).__init__(message)
        self.data_path = data_path
        self.message = message

    def __str__(self):
        return f'Error found in the dataset located at \'{self.data_path}\'. {self.message}'


class NoDataFoundException(IgnnitionException):
    """Exception raised for formatting errors in datasets

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs
        message -- explanation of the exception
    """

    def __init__(self, message):
        super(NoDataFoundException, self).__init__(message)
        self.message = message

    def __str__(self):
        return f'{self.message}'


class DatasetFormatException(DatasetException):
    """Exception raised for formatting errors in datasets

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs
        message -- explanation of the exception
    """

    def __init__(self, data_path, message):
        super(DatasetFormatException, self).__init__(data_path, message)
        self.data_path = data_path
        self.message = message


class TarFileException(DatasetException):
    """Exception raised for formatting errors in datasets

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs
        message -- explanation of the exception
    """

    def __init__(self, data_path, filename, message):
        super(TarFileException, self).__init__(data_path, message)
        self.data_path = data_path
        self.filename = filename
        self.message = message

    def __str__(self):
        return f'Error found in the tar file {self.filename}. {self.message}'


class DatasetNodeException(DatasetFormatException):
    """Exception raised when the path of the dataset is not found

        Attributes:
            dataset -- dataset (training, validation or predict) where the exception occurs
            path -- path of the dataset that raises the exception
            message -- explanation of the exception
        """

    def __init__(self, node_name, data_path, message):
        super(DatasetNodeException, self).__init__(data_path, message)
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
        super(DatasetNotFoundException, self).__init__(message, dataset)
        self.dataset = dataset
        self.message = message
        self.path = path

    def __str__(self):
        return f'Error found in the {self.dataset} dataset located at {self.path}. {self.message}'


class AdditionalFunctionNotFoundException(IgnnitionException):
    """Exception raised when the path of the dataset is not found

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs
        path -- path of the dataset that raises the exception
        message -- explanation of the exception
    """

    def __init__(self, path, message):
        super(AdditionalFunctionNotFoundException, self).__init__(message)
        self.message = message
        self.path = path

    def __str__(self):
        return f'Could not find the additional functions file at {self.path}. {self.message}'


class ModelDescriptionException(IgnnitionException):
    """Exception raised for errors in the model_description.yaml file.

    Attributes:
        dataset -- dataset (training, validation or predict) where the exception occurs.
        message -- explanation of the error
    """

    def __init__(self, message):
        super(ModelDescriptionException, self).__init__(message)
        self.message = message


class FeatureException(IgnnitionException):
    """Exception raised for errors in the model_description.yaml file.

        Attributes:
            feature -- feature name that raises the error
            message -- explanation of the error
    """

    def __init__(self, feature, message):
        super(FeatureException, self).__init__(message)
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
        super(KeywordException, self).__init__(message)
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
        super(EntityError, self).__init__(message)
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
        super(YAMLFormatError, self).__init__(message)
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
        super(YAMLNotFoundError, self).__init__(message)
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
        super(KerasError, self).__init__(message)
        self.variable = variable
        self.parameter = parameter
        self.message = message

    def __str__(self):
        return f'Could not convert the parameter \'{self.parameter}\' into a valid tf.keras \'{self.variable}\'. ' \
               f'{self.message} '


class CombinedAggregationError(IgnnitionException):

    def __init__(self, message=None):
        super(CombinedAggregationError, self).__init__(message)
        self.message = message

    def __str__(self):
        return f'{self.message}'


class OperationError(IgnnitionException):

    def __init__(self, operation, message=None):
        super(OperationError, self).__init__(message)
        self.message = message
        self.operation = operation

    def __str__(self):
        return f'There was an error while computing the operation {self.operation}. {self.message}'


class ProductOperationError(OperationError):

    def __init__(self, operation, prod_type, a, b, message=None):
        super(ProductOperationError, self).__init__(operation, message)
        self.operation = operation
        self.prod_type = prod_type
        self.a = a
        self.b = b
        self.message = message

    def __str__(self):
        return f'There was an error while computing the {self.operation} operation of type {self.prod_type} between ' \
               f'\'{self.a}\' and \'{self.b}\'. {self.message}'


class ConcatOperationError(OperationError):

    def __init__(self, operation, axis, message=None):
        super(ConcatOperationError, self).__init__(operation, message)
        self.operation = operation
        self.axis = axis
        self.message = message

    def __str__(self):
        return f'There was an error while computing the {self.operation} operation using axis = {self.axis}. ' \
               f'{self.message}'


class ConvolutionOperationError(OperationError):

    def __init__(self, operation, message=None):
        super(ConvolutionOperationError, self).__init__(operation, message)
        self.operation = operation
        self.message = message

    def __str__(self):
        return f'There was an error while computing the {self.operation} operation. ' \
               f'{self.message}'


class LossFunctionException(IgnnitionException):

    def __init__(self, loss, message=None):
        super(LossFunctionException, self).__init__(message)
        self.loss = loss
        self.message = message

    def __str__(self):
        return f'Error with the {self.loss} defined as loss. ' \
               f'{self.message}'

class NeuralNetworkNameException(IgnnitionException):

    def __init__(self, name, message=None):
        super(NeuralNetworkNameException, self).__init__(message)
        self.name = name
        self.message = message

    def __str__(self):
        return f'There was an error with the NN named \'{self.name}\'. {self.message}'

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
            FAIL = '\033[91m'
            ENDC = '\033[0m'
            expect_error = str(e)
            try:
                expect_error = expect_error[:expect_error.index("Traceback")]
            except ValueError:
                pass
            print(FAIL + expect_error + ENDC)
            sys.exit(1)

    return wrapper
