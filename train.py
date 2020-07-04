import sys
import torch
import torch.optim as optim
from torch.optim import Adam

import load_data
from universe import InterpretedLanguage
from models import myLSTM

import unicodedata
import string

import numpy as np
import os
import shutil
import random

import pickle
from collections import defaultdict
from statistics import mean

def get_accuracy(truth, prediction):
    """Calculates and returns the similarity of the input lists.
    
    Args:
        truth (list): the ground truth of the original input. 
        prediction (list): the y-hat, the output of the LSTM.
    
    Returns:
        float: The accuracy of the LSTM.
    """
    # Ensure that both lists have the same length
    assert len(truth) == len(prediction)
    correct = 0
    for i in range(len(truth)):

        # Check if elements are identical, increase correct count if they are
        if truth[i] == prediction[i]:
            correct += 1
    return correct/len(truth)

def train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i):
    """ Trains one epoch of the model. No return value.

    Args:
        model (myLSTM): The to be trained LSTM.
        train_data (list): List of tuples, where the first element is the input, and the second element the ground truth.
        loss_function (torch.nn.modules.loss): The specified loss function. In this case, CrossEntropyLoss.
        optimizer (torch.optim.adam): The specified optimizer. In this case, Adam.
        word_to_ix (dict): A dictionary mapping all the words in train_data to an int as index.
        label_to_ix (dict): A dictionary mapping all the labels in train_data to an int as index.
        i (int): The value of the current epoch.
    """
    model.train()
    avg_loss = 0.0
    count = 0
    truths = []
    predictions = []
    batch_sentence = []


    for sentence, label in train_data:
        # Add current idexed label to the ground truth list.
        truths.append(label_to_ix[label])
        
        # Create new hidden layer, detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        
        # Turn both the sentence and the ground truth into a vector using the indices specified.
        sentence = load_data.prepare_sequence(sentence, word_to_ix)
        label = load_data.prepare_label(label, label_to_ix)
        
        # Predict output using the model, save prediction to list.
        prediction = model(sentence)
        prediction_label = prediction.data.max(1)[1]
        predictions.append(int(prediction_label))

        # Calculate loss, save to average loss, optimize model based on loss. 
        model.zero_grad()
        loss = loss_function(prediction, label)
        avg_loss += loss.item()
        count += 1
        if count % 500 == 0:
            print("Epoch: {} Iterations: {} Loss :{}".format(i, count, loss.data[0]))

        loss.backward()
        optimizer.step()

    # Calculate average loss and print this and the accuracy of this epoch.
    avg_loss /= len(train_data)

    print("Epoch {} done! \n train avg_loss: {}, acc: {}\n\n".format(i, avg_loss, get_accuracy(truths,predictions)))


def evaluate(model, data, loss_function, word_to_ix, label_to_ix, name="val"):
    """ Evaluates the model. Returns the accuracy.

    Args:
        model (myLSTM): The to be trained LSTM.
        data (list): List of tuples, where the first element is the input, and the second element the ground truth.
        loss_function (torch.nn.modules.loss): The specified loss function. In this case, CrossEntropyLoss.
        optimizer (torch.optim.adam): The specified optimizer. In this case, Adam.
        word_to_ix (dict): A dictionary mapping all the words in train_data to an int as index.
        label_to_ix (dict): A dictionary mapping all the labels in train_data to an int as index.
        name (string): The type of evaluation, test, train or val, val being standard.
    
    Returns:
        acc (float): The accuracy of the model for this current dataset.  
    """
    model.eval()
    avg_loss = 0.0
    truths = []
    predictions = []
    
    for sentence, label in data:
        # Add current idexed label to the ground truth list.
        truths.append(label_to_ix[label])

        # Create new hidden layer, detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Turn both the sentence and the ground truth into a vector using the indices specified.
        sentence = load_data.prepare_sequence(sentence, word_to_ix)
        label = load_data.prepare_label(label, label_to_ix)

        # Predict output using the model, save prediction to list.
        prediction = model(sentence)
        prediction_label = prediction.data.max(1)[1]
        predictions.append(int(prediction_label))

        # Calculate loss and add it to the total loss value
        loss = loss_function(prediction, label)
        avg_loss += loss.item()
    
    # Calculate and print average loss and accuracy.
    avg_loss /= len(data)
    acc = get_accuracy(truths, predictions)
    print(name + " average loss: {}; accuracy: {}".format(avg_loss, acc)) 
    return acc


hidden = 256
def train():
    """ Creates, trains and evaluates the LSTM. No input nor return value. """
    # Generate and load the data required.
    train_data, val_data, test_data, word_to_ix, label_to_ix, complexity = load_data.load_MR_data()

    # Set all constants required for LSTM and optimizer.
    EMBEDDING_DIM = 256
    HIDDEN_DIM = hidden
    BATCH_SIZE = 1
    BIDIRECTIONAL = "bidirectional" in sys.argv
    VOCAB_SIZE = len(word_to_ix)
    LABEL_SIZE = len(label_to_ix)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 9e-3

    best_val_acc = 0.0

    # Create LSTM
    model = myLSTM(embedding_dim=EMBEDDING_DIM, 
                    hidden_dim=HIDDEN_DIM,
                    vocab_size=VOCAB_SIZE,
                    label_size=LABEL_SIZE,
                    num_layers=1, bidirectional=BIDIRECTIONAL)
    
    # Create optimizer and set loss function
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()


    no_up = 0
    test_lengths = [len(x[0]) for x in train_data]

    def curriculum(i):
        """Allows for curriculum learning, returns max complexity allowed in this EPOCH.
        
        Args:
            i (int): i of current iteration in for-loop. 
        
        Returns:
            float: The maximum complexity allowed in this EPOCH.
        """
        if i < EPOCH:
        #    return 4+2*complexity*i/EPOCH
            return 2+2*i/10
        else:
            return 99999
    
    for i in range(0, EPOCH):
        # Using the curriculum function, this generates the training data for 
        # this epoch based on the max complexity, so that the complexity 
        # gradually increases
        train_data_filtered = [x for x in train_data if len(x[0]) < curriculum(i)]
        
        print("Epoch: {} start.".format(i))

        # Train one epoch
        train_epoch(model, train_data_filtered, criterion, optimizer, word_to_ix, label_to_ix, i)

        # Run current model on validation and test set
        print("Current best validation accuracy:", best_val_acc)
        val_acc = evaluate(model, test_data, criterion, word_to_ix, label_to_ix)
        test_acc = evaluate(model, test_data, criterion, word_to_ix, label_to_ix, "test")
        print("Test accuracy:", test_acc)
        epoch_test_accuracies[nth_run][i] = test_acc
        # Save model if validation accuracy has increased
        if val_acc > best_val_acc:
            print("New best validation accuracy, from {} to {}.".format(best_val_acc, val_acc))
            best_val_acc = val_acc
            os.system("rm " + save_path + "mr_best_model_acc_*.model")
            torch.save(model.state_dict(), save_path + "mr_best_model_acc_{}.model".format(num_indivs, test_acc*10000))
            no_up = 0
        else:
            no_up += 1
            # Stop training if the validation accuracy has not increased for 
            # 22 consecutive EPOCHS, to prevent overfitting.
            if no_up >= 22:
                break
        
    def statsbysize(dataset, name):
        """ Prints the accuracy of the model when ordering input by size.

        Args:
            dataset (list): List of tuples, where the first element is the input, and the second element the ground truth.
            name (string): The name of the dataset, such as train or test.
        """
        bysize = defaultdict(set)
        for i in dataset:
            bysize[len(i[0])].add(i)
        for c in sorted(bysize.keys()):
            print("length {}: {}".format(c, evaluate(model, bysize[c], criterion, word_to_ix, label_to_ix, name)))
    
    print("Training accuracies by size:")
    statsbysize(train_data, "train")
    print("Test accuracies by size:")
    statsbysize(test_data, "test")
    print("Overall test accuracy: {}".format(test_acc))
    
    # Add test accuracy of this model to the list with the test accuracies of other runs of train().
    n_run_test_accuracies.append((s, test_acc))

# Create path-string where model and results will be saved, and ensures that path exists on computer.
num_indivs = int(sys.argv[1])*2
save_path = "best_models_{}_indiv/".format(num_indivs)
pickle_path = "test_accuracies_{}_indiv/"
if not os.path.isdir(save_path):
    os.makedirs(save_path)

if not os.path.isdir(pickle_path):
    os.makedirs(pickle_path)

runs = 10
EPOCH = 100

n_run_test_accuracies = []
epoch_test_accuracies = np.zeros((runs, EPOCH))
nth_run = 0
seeds = []

# Generate different seeds for each run of train()
for i in range(runs):
    seeds.append(random.randint(0, 4294967296))

# Create and train a new model using each seed
for s in seeds:
    torch.manual_seed(s)
    random.seed(s)
    train()
    nth_run += 1

# Calculate the average accuracy over all the runs, and save all 
# individual accuracies to a .tsv file with their respective seeds.
average_test_acc=0.0
with open("accuracies_" + "_".join(sys.argv[1:]) + "_{}dim.tsv".format(hidden), 'w') as o:
    for n in n_run_test_accuracies:
        o.write(str(n[0])+"\t"+str(n[1])+"\n")
        average_test_acc+=n[1]

average_test_acc /= runs

# Save the test accuracies over time using pickle. 
pickle.dump(losses, open(pickle_path  + "accuracy.p", "wb"))

print("Done. ")
print("Average test accuracy = {}".format(average_test_acc))