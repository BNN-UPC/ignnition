import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

def main():
    X = ["50", "100","200", "300", "400"]
    X_axis = np.arange(len(X))
  
    nlinks_50 = []
    nlinks_100 = []
    nlinks_200 = []
    nlinks_300 = []
    nlinks_400 = []

    for filename in os.listdir("./data"):
        if filename.endswith(".npy") :
            if filename=="nlinks_200.npy":
                nlinks_200 = np.load("./data/"+filename)-100
            elif filename=="nlinks_50.npy":
                nlinks_50 = np.load("./data/"+filename)-100
            elif filename=="nlinks_300.npy":
                nlinks_300 = np.load("./data/"+filename)-100
            elif filename=="nlinks_100.npy":
                nlinks_100 = np.load("./data/"+filename)-100
            elif filename=="nlinks_400.npy":
                nlinks_400 = np.load("./data/"+filename)-100
    
    plt.rcParams.update({'font.size': 13})
    plt.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots()
    
    plt.bar(X_axis, [np.mean(nlinks_50), np.mean(nlinks_100), np.mean(nlinks_200), np.mean(nlinks_300), np.mean(nlinks_400)], color = "skyblue", edgecolor='black')

    plt.ylim((0, 14))
    plt.xticks(X_axis, X)
    plt.ylabel('Difference WCGCN w.r.t. WMMSE (%)', fontsize=15)
    plt.xlabel("Size (number of links)", fontsize=15)
    plt.grid(color='gray', axis="y")
    plt.tight_layout()
    plt.savefig('Barplot_figure.pdf', bbox_inches='tight')

if __name__ == "__main__":
    main()