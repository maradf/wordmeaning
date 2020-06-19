import torch 
import torch.optim as optim
from torch.optim import Adam

import one_hot as oh
from universe import InterpretedLanguage

from models import myLSTM

import unicodedata
import string

import numpy as np
import os
import shutil
import random 

import pickle

all_letters = string.ascii_lowercase
n_letters = len(all_letters)

learningrate = 0.01


world_size = 100
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

EMBEDDING_DIM = 256
HIDDEN_DIM = len(world.char2idx)
batch_size = 1
NUM_TARGETS = len(world.names) # len(set([l.item() for l in y_train])) # excluding some 

model = myLSTM(len(world.char2idx), EMBEDDING_DIM, HIDDEN_DIM, batch_size, NUM_TARGETS)
# model.load_state_dict(torch.load("Models_100_indiv/LSTM_iteration_52100.pt"))
# lstm.eval()

# model = myLSTM(embedding_dim=EMBEDDING_DIM, hidden_dim = HIDDEN_DIM, vocab_size=len(char2idx), label_size=len(indiv2idx), num_layers=1)
optimizer = Adam(model.parameters(), lr=learningrate)
criterion = torch.nn.CrossEntropyLoss()

losses = []
val_losses = []
train_acc = []
val_acc = []
nth_iter = 0
pickle_path = "Results_{}_indiv/".format(world_size)
model_path = "Models_{}_indiv/".format(world_size)

if not os.path.isdir(pickle_path):
    os.makedirs(pickle_path)

if not os.path.isdir(model_path):
    os.makedirs(model_path)

pickle.dump(y_train, open(pickle_path  + "y_train.p", "wb"))
pickle.dump(y_train, open(pickle_path  + "y_val.p", "wb"))
curr_learning_x = world.model_input(world.names, "in")
curr_learning_y = world.model_input(world.names, "out")
num_loops = 200
begin = 0
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
        train_out = model(x.unsqueeze(0))
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
            save_path = model_path + "LSTM_iteration_{}.pt".format(nth_iter)
            torch.save(model.state_dict(), save_path)
            pickle.dump(losses, open(pickle_path  + "losses.p", "wb"))
            pickle.dump(train_out.argmax(axis=1), open(pickle_path  + "train_out.p", "wb"))
        nth_iter += 1

        if train_out.argmax(axis=1) == y:
            correct += 1
        
    total = len(X_train)
    train_acc.append(correct / total)
    pickle.dump(train_acc, open(pickle_path  + "train_acc.p", "wb"))    
    losses.append(loss.item())
    
    for i in range(len(X_val)):
        model.eval()
        val_x = torch.LongTensor(X_val[i])
        val_y = torch.LongTensor(y_val[i])
        val_out = model(val_x.unsqueeze(0))
        val_out = val_out[:, -1, :]
        val_loss = criterion(val_out, val_y)
        val_losses.append(val_loss)
    
        pickle.dump(val_losses, open(pickle_path  + "val_losses.p", "wb"))
        pickle.dump(val_out.argmax(axis=1), open(pickle_path  + "val_out.p", "wb"))
        #function accuracy 
        if val_out.argmax(axis=1) == val_y:
            val_correct += 1
    total = len(X_val)
    val_acc.append(val_correct / total)
    pickle.dump(val_acc, open(pickle_path  + "val_acc.p", "wb"))

torch.save(model.state_dict(), model_path + "Last_Model.pt")