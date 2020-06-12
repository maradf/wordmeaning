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

all_letters = string.ascii_lowercase
n_letters = len(all_letters)

n_training_runs = 1000
learningrate = 0.01


world_size = 4
n_pairs = int(world_size/2)
# batch_size = 2

world = InterpretedLanguage(rel_num=4, num_pairs = n_pairs)
all_relations = world.allexamples(b="l")
X_train, y_train, X_val, y_val, X_test, y_test = world.dataset("l")
all_individuals = world.reserved_chars
X_train_lengths = world.sentence_lengths(X_train)
X_val_lengths = world.sentence_lengths(X_val)
X_test_lengths = world.sentence_lengths(X_test)
print(X_test)
print(X_test_lengths)
vec_X_train = world.model_input(X_train, 5)
vec_y_train = world.model_input(y_train, 0)
vec_X_val = world.model_input(X_val, 5)
vec_y_val = world.model_input(y_val, 0)
vec_X_test = world.model_input(X_test, 5)
vec_y_test = world.model_input(y_test, 0)
indiv2idx = world.indiv2idx
char2idx = world.char2idx



lstm = myLSTM(nb_layers=8, indiv2idx=indiv2idx, char2idx=char2idx, batch_size=1)
y_hat = lstm.forward(vec_X_train, X_train_lengths)
print("one_hot", vec_y_train)
print("forward output", y_hat)
# optimizer = Adam(lstm.parameters(), lr=learningrate)
# fifty_losses = []
# for i in range(n_training_runs):
#     lstm.train()
#     y_hat = lstm.forward(vec_X_train, X_train_lengths)
#     loss = lstm.loss(y_hat, vec_y_train)
    
#     float_loss = round(float(loss), 2)
#     fifty_losses.append(float_loss)
#     if i % 50 == 0:
#         average_loss = np.average(fifty_losses)
#         print("Iteration: {}% {}/{}".format(round((i/n_training_runs)*100, 2), i+1, n_training_runs))
#         print("Loss: {}".format(average_loss))
#         save_path = "Model_iteration_{}.pt".format(i)
#         torch.save(lstm.state_dict(), save_path)

#         # Save loss value to separate file
#         lossf = open(loss_path, "a+")
#         lossf.write("{}\n".format(average_loss))
#         lossf.close()
#         fifty_losses = []
#     optimizer.zero_grad()
#     loss.backward()
    # optimizer.step()
# print(Y_hat)