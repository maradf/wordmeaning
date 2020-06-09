import numpy as np

import universe as uni


world_size = 4
current_world = uni.newUniverse(world_size)
print("world = ", current_world)

test = uni.InterpretedLanguage(rel_num=4, num_pairs=6)

example = test.examples(4)

print(test.allexamples(b='l'))

all_examples = test.allexamples(b="l")

def one_hot_encoding(inputs):
    individuals = []
    for elem in inputs:
        if elem[1] not in individuals:
            individuals.append(elem[1])

    one_hot = np.zeros((len(inputs), len(individuals)))
    

    for i in range(len(inputs)):
        j = individuals.index(inputs[i][1])
        one_hot[i][j] = 1
    return one_hot

print(one_hot_encoding(all_examples))

