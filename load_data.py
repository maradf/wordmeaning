"""
Written by: Dr. Denis Paperno
Edited and documented by: Mara Fennema

Code to create and load the dataset required for train.py

Code requires the following command line arguments, in this order:
    number of individual pairs in the model, thus half of the maximum individuals.
    number of relations in the model
    n - maximal complexity of examples
    branching type (l,r,rl)
    (optional) proportion of examples of complexity n included in the training data
"""

import sys
import torch
import torch.autograd as autograd
import random
import torch.utils as Data
import universe

SEED = 501
def prepare_sequence(sequence, to_ix):
    """ Takes a string and returns a tensor of ints representing that string.

    Args:
        sequence (string): String representing a description of a relationship.
        to_ix (dict): Dict with as keys all the unique tokens in the dataset,
                      and as value a corresponding index. 

    Returns:
        new_sequence (torch.LongTensor): Tensor of the same length as the sequence 
                            string, with each int in the string representing each char.
    """
    new_sequence = autograd.Variable(torch.LongTensor([to_ix[i] for i in sequence]))
    return new_sequence

def prepare_label(label,label_to_ix, cuda=False):
    """ Takes a string and returns a tensor containing an int 
        representing that string.

    Args:
        label (string): String representing a single individual.
        to_ix (dict): Dictionary with all possible individuals in the label
                      string as keys and an int as their index as value.

    Returns:
        var (torch.LongTensor): Tensor containing the int representing the label.
    """
    var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
    return var

def build_token_to_ix(sentences):
    """ Create a dict with all possible tokens in the sentences as keys
        with corresponding indices as values.

    Args: 
        sentences (list): List containing all sentences in the dataset.
    
    Returns:
        token_to_ix (dict): Dict with as keys all the unique tokens in the 
                            dataset, and as value a corresponding index, 
                            including one padding token. 
    """
    token_to_ix = dict()
    print("There are a total of {} strings in the dataset.".format(len(sentences)))
    
    # Check for each token in each sentence if it is already present as a key
    # in the token_to_ix dict, if not, it is added, with the current length
    # as value. 
    for sent in sentences:
        for token in sent:
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    
    # Padding token is added for if batches are used.
    token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix

def build_label_to_ix(labels):
    """ Create a dict with all individuals in dataset as keys
        with their corresponding indices as value.

    Args: 
        labels (list): List containing all individuals in the dataset.
    
    Returns:
        label_to_ix (dict): Dict with as keys all the unique labels in the 
                            dataset, and as value a corresponding index.
    """
    label_to_ix = dict()
    
    # Check for each label available if it is present in the dict, if it is not
    # it gets added, with the current length of the dict as value. 
    for label in labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)

def load_MR_data():
    """ Create load and return the dataset, split into train, val and test sets.
        Also returns two indexing dicts and the maximum complexity used.

        Returns:
            train_data (list): The created training data. List of tuples with 
                               each first argument being the input, and each 
                               second being the ground truth. 
            val_data (list): The created validation data. List of tuples with
                             each first argument being the input, and each 
                             second being the ground truth.
            test_data (list): The created test data. List of tuples with 
                              each first argument being the input, and each 
                              second being the ground truth.
            word_to_ix (dict): Dict with as keys all the unique tokens in the 
                               dataset, and as value a corresponding index.
            label_to_ix (dict): Dict with as keys all the unique labels in the 
                                dataset, and as value a corresponding index.
            complexity (int): Maximum complexity of the sentences in the data.
    """

    print('Generating Data...')

    # Create universe with specified parameters in the user input.
    num_pairs=int(sys.argv[1])
    rel_num=int(sys.argv[2])
    L=universe.InterpretedLanguage(rel_num,num_pairs)
    k=num_pairs*5*rel_num
    branching=sys.argv[4]
    complexity=int(sys.argv[3])
    
    # Get all possible relationships with a complexity one smaller than
    # the maximum defined complexity. The trainingset will be created 
    # using this.
    thedata = L.allexamples(branching,complexity=complexity-1)
    random.seed(SEED)
    random.shuffle(thedata)
    
    # Get all the possible relationships with only the maximum complexity.
    # Part of the training set, and the entire test set and the validationset
    # will be created using this.
    devtest=L.allexamples(branching,complexity=complexity,min_complexity=complexity)
    random.shuffle(devtest)
    datasize=len(devtest)
    
    # The trainingset uses 80% of all examples using the maximum complexity
    # and the entirety of the dataset which has the maximum complexity - 1.
    p=0.8
    train_data = thedata+devtest[:int(datasize*p)]

    # The validationset uses 11% of the examples using the maximum complexity.
    val_data = devtest[int(datasize*p):int(datasize*(p+(1-p)*0.55))]

    # The testset uses the remaining 9% of all the examples with
    # the maximum complexity.
    test_data = devtest[int(datasize*(p+(1-p)*0.55)):]
    print("Total data: {}\nTrain: {}\nVal: {}\nTest: {}".format(datasize+len(thedata),len(train_data),len(val_data),len(test_data)))

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Create both the word-to-index dict and the label-to-index dict.
    word_to_ix = build_token_to_ix([s for s,_ in train_data+val_data+test_data])
    label_to_ix = {val:idx for idx,val in enumerate(L.names)}
    print("word_to_ix = ", word_to_ix)
    print("label_to_ix = ", label_to_ix)
    print('vocab size:',len(word_to_ix),'label size:',len(label_to_ix))
    print("Data generation done.")

    return train_data, val_data, test_data, word_to_ix, label_to_ix, complexity

