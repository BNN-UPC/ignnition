import shutil
import yaml
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ignnition

__test__ = "Routenet"


def _copy_and_replace_files():
    model_config = Path().parent / "examples" / __test__
    p = Path(__file__).parent / __test__
    try:
        shutil.copytree(model_config.absolute(), p)
    except FileExistsError:
        # copy it again to make sure that it is the latest version
        shutil.rmtree(p)
        shutil.copytree(model_config.absolute(), p)
    train_options = p / "train_options.yaml"
    with open(train_options, "r") as f:
        options = yaml.safe_load(f.read())
    options["epochs"] = 2
    options["epoch_size"] = 1000
    options["shuffle_training_set"] = False
    for key, val in options.items():
        if isinstance(options[key], str) and options[key].startswith("./"):
            options[key] = options[key].replace("./", f"{model_config.absolute()}/")

    if "predict_dataset" not in options:
        options["predict_dataset"] = options["validation_dataset"]

    with train_options.open("w") as f:
        yaml.dump(options, f)


def _clean_files():
    p = Path(__file__).parent / __test__
    shutil.rmtree(p)


def main():
    _copy_and_replace_files()
    model = ignnition.create_model(model_dir=Path(__file__).parent / __test__)
    model.computational_graph()
    model.train_and_validate()
    model.predict()
    _clean_files()


if __name__ == "__main__":
    main()
