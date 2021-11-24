from ignnition._version import __version__
from ignnition.ignnition_model import IgnnitionModel
from ignnition.error_handling import handle_exception


@handle_exception
def create_model(model_dir):
    """
    This method creates and returs an IGNNITION model which serves as interface to the whole ignnition framework

    Parameters
    ----------
    model_dir : str
        Path to the directory where the model_description, global_variables and train_options.yaml are found
    """
    return IgnnitionModel(model_dir)
