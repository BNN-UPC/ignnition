import ignnition
from pathlib import Path


def main():
    model = ignnition.create_model(model_dir=Path(__file__).parent.absolute())
    model.computational_graph()
    model.train_and_validate()
    # model.predict()


if __name__ == "__main__":
    main()
