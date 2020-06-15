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

n_training_runs = 1000
learningrate = 0.01


world_size = 10
n_pairs = int(world_size/2)
# batch_size = 2



def train_model(model, X_train, y_train, epochs=10):
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        x = X_train.long()
        y = y_train.long()
        y_pred = model(x)
        y_pred = y_pred.reshape([52, 1])

        #optimizer.zero_grad()
        loss = criterion(out, torch.max(y_train, 1)[1])

        loss.backward(retain_graph=True)
        #optimizer.step()
        sum_loss += loss.item()*y.shape[0]
        total += y.shape[0]
        val_loss, val_acc = validation_metrics(model, val_x, val_y)
        if i % 5 == 1:
            print("train loss {}, val loss {}, val accuracy {}".format(sum_loss/total, val_loss, val_acc))


def validation_metrics (model, val_x, val_y):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    x = val_x.long()
    y = val_y.long()
    y_hat = model(x)
    y_hat = y_hat.reshape([14, 1])
    loss = criterion(y_hat, torch.max(y, 1)[1])
    pred = torch.max(y_hat, 1)[1]
    correct += (pred == y).float().sum()
    total += y.shape[0]
    sum_loss += loss.item()*y.shape[0]
    #sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total


world = InterpretedLanguage(rel_num=4, num_pairs = n_pairs)
all_relations = world.allexamples(b="l")

X_train, y_train, X_val, y_val, X_test, y_test = world.dataset("l")
all_individuals = world.names
X_train_lengths = world.sentence_lengths(X_train)
X_val_lengths = world.sentence_lengths(X_val)
X_test_lengths = world.sentence_lengths(X_test)
# print(X_train)
# print(X_train_lengths)
vec_X_train = world.model_input(X_train, 5, "in")
vec_y_train = world.model_input(y_train, 0, "out")
vec_X_val = world.model_input(X_val, 5, "in")
vec_y_val = world.model_input(y_val, 0, "out")
vec_X_test = world.model_input(X_test, 5, "in")
vec_y_test = world.model_input(y_test, 0, "out")
indiv2idx = world.indiv2idx
char2idx = world.char2idx



HIDDEN_DIM = len(world.char2idx)
EMBEDDING_DIM = 256

model = myLSTM(embedding_dim=EMBEDDING_DIM, hidden_dim = HIDDEN_DIM, vocab_size=len(char2idx), label_size=len(indiv2idx), num_layers=1)
optimizer = Adam(model.parameters(), lr=learningrate)
criterion = torch.nn.CrossEntropyLoss()

losses = []
val_losses = []
train_acc = []
val_acc = []
nth_iter = 0
pickle.dump(vec_y_train, open("Results_10_indiv/vec_y_train.p", "wb"))
pickle.dump(vec_y_train, open("Results_10_indiv/vec_y_val.p", "wb"))

for epoch in range(200):  
    # random.shuffle(vec_X_train)
    # print(vec_y_train)
    for i in range(len(vec_X_train)):
        model.train()
        train_out = model(vec_X_train[i])
        train_out2 = train_out[:, -1] # pick the last output vecs. of the sequence for every example in the batch
        loss = criterion(train_out, vec_y_train[i])
        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True)
        optimizer.step()
    
        if (nth_iter) % 50 == 0:
            print("Iteration {} out of {}".format(nth_iter, 200*(len(vec_X_train))))
            print("Loss\t {}".format(loss.item()))
            save_path = "Models/LSTM_iteration_{}.pt".format(nth_iter)
            torch.save(model.state_dict(), save_path)
            pickle.dump(losses, open("Results_10_indiv/losses.p", "wb"))
            pickle.dump(train_out.argmax(axis=1), open("Results_10_indiv/train_out.p", "wb"))
        nth_iter += 1

        
    correct = 0
    for pred, true in zip(train_out.argmax(axis=1), vec_y_train):
        if pred == true:
            correct += 1
        
    total = len(train_out)
    train_acc.append(correct / total)
    pickle.dump(train_acc, open("Results_10_indiv/train_acc.p", "wb"))
    
    losses.append(loss.item())
    
    for i in range(len(vec_X_val)):
        model.eval()
        val_out = model(vec_X_val[i])
        # val_out = val_out[:, -1, :]
        val_loss = criterion(val_out, vec_y_val[i])
        val_losses.append(val_loss)
        correct = 0
    
    pickle.dump(val_losses, open("Results_10_indiv/val_losses.p", "wb"))
    pickle.dump(val_out.argmax(axis=1), open("Results_10_indiv/val_out.p", "wb"))
    #function accuracy 
    for pred, true in zip(val_out.argmax(axis=1), vec_y_val):
        if pred == true:
            correct += 1
     
    total = len(train_out)
    val_acc.append(correct / total)
    pickle.dump(val_acc, open("Results_10_indiv/val_acc.p", "wb"))



# def train_model(model, X_train, y_train, epochs=10):
#     #parameters = filter(lambda p: p.requires_grad, model.parameters())
#     #optimizer = torch.optim.Adam(parameters, lr=lr)
#     for i in range(epochs):
#         model.train()
#         sum_loss = 0.0
#         total = 0
        
#         x = X_train.long()
#         y = y_train.long()
#         y_pred = model(x)
#         y_pred = y_pred.reshape([52, 1])


#         #optimizer.zero_grad()
#         loss = criterion(out, torch.max(y_train, 1)[1])


#         loss.backward(retain_graph=True)
#         #optimizer.step()
#         sum_loss += loss.item()*y.shape[0]
#         total += y.shape[0]
#         val_loss, val_acc = validation_metrics(model, val_x, val_y)
#         if i % 5 == 1:
#             print("train loss {}, val loss {}, val accuracy {}".format(sum_loss/total, val_loss, val_acc))

# def validation_metrics (model, val_x, val_y):
#     model.eval()
#     correct = 0
#     total = 0
#     sum_loss = 0.0
#     sum_rmse = 0.0
#     x = val_x.long()
#     y = val_y.long()
#     y_hat = model(x)
#     y_hat = y_hat.reshape([14, 1])
#     loss = criterion(y_hat, torch.max(y, 1)[1])
#     pred = torch.max(y_hat, 1)[1]
#     correct += (pred == y).float().sum()
#     total += y.shape[0]
#     sum_loss += loss.item()*y.shape[0]
#     #sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
#     return sum_loss/total, correct/total




# for epoch in range(200):
#     random.shuffle(vec_X_train)
#     model.train()
#     avg_loss = 0.0
#     nth_run = 0
#     for sentence in vec_X_train:
#         model.hidden = model.init_hidden()
        


# y_hat = model.forward(vec_X_train[0])
# for i in range(len(vec_X_train)):
#     x = vec_X_train[i]
#     print(x)
#     y = torch.tensor(X_train_lengths[i])
#     y_hat = model.forward(x, y)

# y_hat = model.forward(vec_X_train, X_train_lengths)
# print("one_hot", vec_y_train)
# print("forward output", y_hat)


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