import torch 
import one_hot as oh
import universe as uni

import unicodedata
import string

all_letters = string.ascii_lowercase
n_letters = len(all_letters)



world_size = 4
n_pairs = int(world_size/2)

world = uni.InterpretedLanguage(rel_num=4, num_pairs = n_pairs)
all_relations = world.allexamples(b="l")
all_individuals = []
all_lefthand_sides = []
relations_per_idividuals = dict()

for i in range(world_size):
    all_individuals.append(all_relations[i][0])

for elem in all_relations:
    all_lefthand_sides.append(elem[0])
    if elem[1] in relations_per_idividuals.keys():
        relations_per_idividuals[elem[1]].append(elem[0])
    else:
        relations_per_idividuals[elem[1]] = [elem[0]]

print(relations_per_idividuals)

print(all_lefthand_sides)

print(all_individuals)
y = oh.one_hot_golden_standard(all_relations)

print(lineToTensor("asp").size())
# # print("y_hat: ", y_hat)
# criterion = torch.nn.CrossEntropyLoss()
# criterion(y_hat, y.argmax())
