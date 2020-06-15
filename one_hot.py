import numpy as np

import universe as uni
import string

all_letters = string.ascii_lowercase
n_letters = len(all_letters)


# world_size = 4
# current_world = uni.newUniverse(world_size)
# # print("world = ", current_world)

# test = uni.InterpretedLanguage(rel_num=4, num_pairs=(int(world_size/2)))

# example = test.examples(4)

# print(test.allexamples(b='l'))

# all_examples = test.allexamples(b="l")

def one_hot_golden_standard(inputs):
    individuals = []
    for elem in inputs:
        if elem[1] not in individuals:
            individuals.append(elem[1])

    one_hot = np.zeros((len(inputs), len(individuals)))
    

    for i in range(len(inputs)):
        j = individuals.index(inputs[i][1])
        one_hot[i][j] = 1
    return one_hot


def letterToIndex(letter):
    return all_letters.find(letter)

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# print(one_hot_encoding(all_examples))

