import torch 
import one_hot as oh
from universe import InterpretedLanguage

from models import myLSTM

import unicodedata
import string


all_letters = string.ascii_lowercase
n_letters = len(all_letters)



world_size = 4
n_pairs = int(world_size/2)

world = InterpretedLanguage(rel_num=4, num_pairs = n_pairs)
all_relations = world.allexamples(b="l")
X_train, y_train, X_val, y_val, X_test, y_test = world.dataset("l")
all_individuals = world.reserved_chars
vec_X_train = world.model_input(X_train, 5)
vec_y_train = world.model_input(y_train, 0)
vec_X_val = world.model_input(X_val, 5)
vec_y_val = world.model_input(y_val, 0)
vec_X_test = world.model_input(X_test, 5)
vec_y_test = world.model_input(y_test, 0)
indiv2idx = world.indiv2idx
char2idx = world.char2idx


lstm = myLSTM(nb_layers=8, indiv2idx=indiv2idx, char2idx=char2idx, batch_size=1)
