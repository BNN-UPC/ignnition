from typing import (
    Union,
    List,
    Optional
)
from types import GeneratorType

from ignnition._version import __version__
from ignnition.ignnition_model import IgnnitionModel
from ignnition.error_handling import handle_exception

import networkx as nx


@handle_exception
def create_model(model_dir,
                 training_data: Optional[Union[GeneratorType, List[nx.Graph]]] = None,
                 validation_data: Optional[Union[GeneratorType, List[nx.Graph]]] = None):
    """
    This method creates and returs an IGNNITION model which serves as interface to the whole ignnition framework

    Parameters
    ----------
    model_dir : str
        Path to the directory where the model_description, global_variables and train_options.yaml are found
    training_data:
        Training dataset. Either a generator of networkx graphs or the networkx graphs already loaded
    validation_data:
        Validation dataset. Either a generator of networkx graphs or the networkx graphs already loaded
    """
    return IgnnitionModel(model_dir, training_data, validation_data)
