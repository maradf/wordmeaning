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
"""
world_size = 4
n_pairs = int(world_size/2)
# batch_size = 2


world = InterpretedLanguage(rel_num=4, num_pairs = n_pairs)
all_relations = world.allexamples(b="l")

X_train, y_train, X_val, y_val, X_test, y_test = world.dataset("l")
all_individuals = world.names
X_train_lengths = world.sentence_lengths(X_train)
X_val_lengths = world.sentence_lengths(X_val)
X_test_lengths = world.sentence_lengths(X_test)
X_train = world.model_input(X_train, "in")
y_train = world.model_input(y_train, "out")
X_val = world.model_input(X_val, "in")
y_val = world.model_input(y_val, "out")
X_test = world.model_input(X_test, "in")
y_test = world.model_input(y_test, "out")
indiv2idx = world.indiv2idx
char2idx = world.char2idx

# HIDDEN_DIM = len(world.char2idx)
# EMBEDDING_DIM = 18


losses = []
val_losses = []
train_acc = []
val_acc = []
nth_iter = 0
# pickle_path = "Results_{}_indiv/".format(world_size)
# model_path = "Models_{}_indiv/".format(world_size)

# if not os.path.isdir(pickle_path):
#     os.makedirs(pickle_path)

# if not os.path.isdir(model_path):
#     os.makedirs(model_path)

# pickle.dump(y_train, open(pickle_path  + "y_train.p", "wb"))
# pickle.dump(y_train, open(pickle_path  + "y_val.p", "wb"))
curr_learning_x = world.model_input(world.names, "in")
curr_learning_y = world.model_input(world.names, "out")
num_loops = 200
begin = 0
# print(X_train)

"""
def train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i):
    model.train()
    
    avg_loss = 0.0
    count = 0
    truths = []
    predictions = []
    batch_sentence = []

    for sentence, label in train_data:
        truths.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sentence = load_data.prepare_sequence(sentence, word_to_ix)
        label = load_data.prepare_label(label, label_to_ix)
        prediction = model(sentence)
        prediction_label = prediction.data.max(1)[1]
        predictions.append(int(prediction_label))
        model.zero_grad()
        loss = loss_function(prediction, label)
        avg_loss += loss.item()
        count += 1
        if count % 500 == 0:
            print("Epoch: {} Iterations: {} Loss :{}".format(i, count, loss.data[0]))

        loss.backward()
        optimizer.step()
    avg_loss /= len(train_data)

    print("Epoch {} done! \n train avg_loss: {}, acc: {}\n\n".format(i, avg_loss, get_accuracy(truths,predictions)))

def get_accuracy(truth, prediction):
    assert len(truth) == len(prediction)
    correct = 0
    for i in range(len(truth)):
        if truth[i] == prediction[i]:
            correct += 1
    return correct/len(truth)

def evaluate(model, data, loss_function, word_to_ix, label_to_ix, name="val"):
    model.eval()
    avg_loss = 0.0
    truths = []
    predictions = []
    
    for sentence, label in data:
        truths.append(label_to_ix[label])
        model.hidden = model.init_hidden()
        sentence = load_data.prepare_sequence(sentence, word_to_ix)
        label = load_data.prepare_label(label, label_to_ix)
        prediction = model(sentence)
        prediction_label = prediction.data.max(1)[1]
        predictions.append(int(prediction_label))
        loss = loss_function(prediction, label)
        avg_loss += loss.item()
    avg_loss /= len(data)
    acc = get_accuracy(truths, predictions)
    print(name + " average loss: {}; accuracy: {}".format(avg_loss, acc)) 
    return acc


hidden = 256
def train():
    train_data, val_data, test_data, word_to_ix, label_to_ix, complexity = load_data.load_MR_data()

    EMBEDDING_DIM = 256
    HIDDEN_DIM = hidden
    BATCH_SIZE = 1
    BIDIRECTIONAL = "bidirectional" in sys.argv
    VOCAB_SIZE = len(word_to_ix)
    LABEL_SIZE = len(label_to_ix)
    LEARNING_RATE = 1e-3

    best_val_acc = 0.0

    model = myLSTM(embedding_dim=EMBEDDING_DIM, 
                    hidden_dim=HIDDEN_DIM,
                    vocab_size=VOCAB_SIZE,
                    label_size=LABEL_SIZE,
                    num_layers=1, bidirectional=BIDIRECTIONAL)
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=9e-3)
    criterion = torch.nn.CrossEntropyLoss()

    no_up = 0
    test_lengths = [len(x[0]) for x in train_data]
    print(test_lengths)

    def curriculum(i):
        "Allows for curriculum learning, returns max curriculum allowed in this EPOCH."
        if i < EPOCH:
        #    return 4+2*complexity*i/EPOCH
            return 2+2*i/10
        else:
            return 99999
    
    for i in range(0, EPOCH):
        # Using the curriculum function, this generates the training data for this epoch
        # based on the max complexity, so that the complexity gradually increases
        train_data_filtered = [x for x in train_data if len(x[0]) < curriculum(i)]
        
        print("Epoch: {} start.".format(i))

        # One epoch worth of training
        train_epoch(model, train_data_filtered, criterion, optimizer, word_to_ix, label_to_ix, i)

        print("Current best validation accuracy:", best_val_acc)
        val_acc = evaluate(model, test_data, criterion, word_to_ix, label_to_ix)

        test_acc = evaluate(model, test_data, criterion, word_to_ix, label_to_ix, 'test')
        print("Test accuracy:", test_acc)
        epoch_test_accuracies[nth_run][i] = test_acc
        if val_acc > best_val_acc:
            print("New best validation accuracy, from {} to {}.".format(best_val_acc, val_acc))
            best_val_acc = val_acc
            os.system("rm " + save_path + "mr_best_model_acc_*.model")
            torch.save(model.state_dict(), save_path + "mr_best_model_acc_{}.model".format(num_indivs, test_acc*10000))
            no_up = 0
        else:
            no_up += 1
            if no_up >= 22:
                break
        
    def statsbysize(dataset):
        bysize = defaultdict(set)
        for i in dataset:
            bysize[len(i[0])].add(i)
        for c in sorted(bysize.keys()):
            print("length {}: {}".format(c, evaluate(model, bysize[c], criterion, word_to_ix, label_to_ix, 'train')))
    print("Training accuracies by size:")
    statsbysize(train_data)
    print("Test accuracies by size:")
    statsbysize(test_data)
    print("Overall test accuracy: {}".format(test_acc))
    n_run_test_accuracies.append((s, test_acc))

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



for i in range(runs):
    seeds.append(random.randint(0, 4294967296))

for s in seeds:
    torch.manual_seed(s)
    random.seed(s)
    train()
    nth_run += 1

# print(n_run_test_accuracies)
# test_acc = mean(n_run_test_accuracies)

# print('The average test accuracy of {} runs is {}'.format(runs, test_acc))

test_acc=0.0
with open("accuracies_" + "_".join(sys.argv[1:]) + "_{}dim.tsv".format(hidden), 'w') as o:
    for n in n_run_test_accuracies:
        o.write(str(n[0])+"\t"+str(n[1])+"\n")
        test_acc+=n[1]

# pickle.dump(losses, open(pickle_path  + "accuracy.p", "wb"))

test_acc /= runs
print("Done. ")
print("Average test accuracy = {}".format(test_acc))

# train()

"""
for loop in range(begin, num_loops):  

    correct = 0
    val_correct = 0
    for i in range(len(X_train)):
        model.train()

        if nth_iter < len(curr_learning_x):
            x = torch.LongTensor(curr_learning_x[i])
            y = torch.LongTensor(curr_learning_y[i])
        else:
            x = torch.LongTensor(X_train[i])
            y = torch.LongTensor(y_train[i])
        print(x)
        print(y)
        train_out = model(x)
        print(train_out)
        train_out = train_out[:, -1, :]
        loss = criterion(train_out, y)
        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True)
        optimizer.step()
    
        if (nth_iter) % 500 == 0:
            print("Iteration\t {} out of {}".format(nth_iter+1, num_loops*(len(X_train))))
            print("\t\t {:.2f}%".format(((1 + nth_iter)*100) / (num_loops*(len(X_train)))))
            print("Loss\t\t {}\n".format(loss.item()))
            # save_path = model_path + "LSTM_iteration_{}.pt".format(nth_iter)
            # torch.save(model.state_dict(), save_path)
            # pickle.dump(losses, open(pickle_path  + "losses.p", "wb"))
            # pickle.dump(train_out.argmax(axis=1), open(pickle_path  + "train_out.p", "wb"))
        nth_iter += 1

        if train_out.argmax(axis=1) == y:
            correct += 1
        
    total = len(X_train)
    train_acc.append(correct / total)
    # pickle.dump(train_acc, open(pickle_path  + "train_acc.p", "wb"))    
    losses.append(loss.item())
    
    for i in range(len(X_val)):
        model.eval()
        val_x = torch.LongTensor(X_val[i])
        val_y = torch.LongTensor(y_val[i])
        val_out = model(val_x.unsqueeze(0))
        val_out = val_out[:, -1, :]
        val_loss = criterion(val_out, val_y)
        val_losses.append(val_loss)
    
        # pickle.dump(val_losses, open(pickle_path  + "val_losses.p", "wb"))
        # pickle.dump(val_out.argmax(axis=1), open(pickle_path  + "val_out.p", "wb"))
        #function accuracy 
        if val_out.argmax(axis=1) == val_y:
            val_correct += 1
    total = len(X_val)
    val_acc.append(val_correct / total)
    # pickle.dump(val_acc, open(pickle_path  + "val_acc.p", "wb"))

# torch.save(model.state_dict(), model_path + "Last_Model.pt")
"""