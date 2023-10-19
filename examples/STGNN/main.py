import ignnition
import csv

def main():
    model = ignnition.create_model(model_dir='./')
    model.computational_graph()
    # model.train_and_validate()

    predicts=model.predict()

    with open("results_v2.csv","w") as f:
        write = csv.writer(f)
        for i in predicts:
            ex = i.numpy()
            write.writerow(ex)

if __name__ == "__main__":
    main()
