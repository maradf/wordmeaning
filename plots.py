import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import pandas as pd

def plot_test_accuracies(n_indivs=""):
    path = "Test Accuracies/"
    for folder in os.listdir(path):
        if n_indivs:
            if n_indivs not in folder:
                if os.listdir(path).index(folder) == len(os.listdir(path)):
                    raise "The chosen integer has no data."
                continue
        filepath = path + folder + "/" + os.listdir(path+folder)[0]
        accs = pickle.load(open(filepath, "rb"))
        for i in range(len(accs)):
            non_zero_index = 0
            for j in range(len(accs[i])):
                if accs[i][j] > 0:
                    non_zero_index = j
            y = accs[i][:non_zero_index]
            x = range(len(y))
            plt.figure(i+1)
            plt.title(filepath)
            plt.plot(x, y)
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
        plt.show()

def plot_final_accuaries():
    path = "Accuracy files/"
    non_bidirectional = []
    bidirectional = []
    for f in os.listdir(path):
        filepath = path + f
        df = pd.read_csv(filepath, header=None, sep='\t')
        accs = [elem for elem in df[1]]
        if "bidirectional" not in filepath:
            non_bidirectional.append(sum(accs)/len(accs))
        else:
            bidirectional.append(sum(accs)/len(accs))
    x = range(4, len(non_bidirectional)*2+4, 2)
    plt.figure(1)
    plt.title("Non-bidirectional accuracies")
    plt.plot(x, non_bidirectional)
    plt.xlabel("Number of individuals")
    plt.ylabel("Accuracy")

    x = range(4, len(bidirectional)*2+4, 2)
    plt.figure(2)
    plt.title("Bidirectional accuracies")
    plt.plot(x, bidirectional)
    plt.xlabel("Number of individuals")
    plt.ylabel("Accuracy")
    plt.show()

plot_test_accuracies("_4_")
# plot_final_accuaries()