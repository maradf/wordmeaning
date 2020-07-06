"""
Written by: Mara Fennema

Contains functions to plot the all the accuracies saved to files by train.py.
"""

import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import pandas as pd

def plot_test_accuracies(path, n_indivs=""):
    """Plots the test accuracies over time of all models of one number of 
    individuals at the same time. Does this both for the bidirectional models
    as the non-bidirectional models.

    Args:
        path (string): The path where the folders with data are found. 
        n_indivs (string): The desired number of individuals, preceded and 
        followed by an underscore.
    
    Example:
        plot_test_accuracies("Test Accuracies/", "_4_")
            This results in the plots of all runs for 4 individuals, both
            the plots of the bidirectional models and the 
            non-bidirectional models, all of which can be found in the 
            folder called Test Accuracies.

    """
    for folder in os.listdir(path):
        
        # Check if this iteration of the loop needs to be skipped for it does not 
        # contain the specified number of individuals.
        if n_indivs:
            if n_indivs not in folder:
                if os.listdir(path).index(folder) == len(os.listdir(path)) - 1:
                    print("The chosen number has no data. Please try another value.")
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

def plot_final_accuaries(path):
    """Creates a plot with the mean accuracy of all final models, to compare
    the different accuracies for the differnt numbers of individuals.

    Args:
        path (string): Path to where all the accuracy files can be found. 

    """
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

# plot_test_accuracies("Test Accuracies/", "_4_")
# plot_final_accuaries("Accuracy files/")