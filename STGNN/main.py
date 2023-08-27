

import ignnition


def main():
    model = ignnition.create_model(model_dir='./')
    print("aixo hauria de ser un model")
    model.computational_graph()

    model.train_and_validate()
    #model.predict()

if __name__ == "__main__":
    main()
