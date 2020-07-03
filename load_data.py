import sys
import torch
import torch.autograd as autograd
import random
import torch.utils as Data
import universe

SEED = 501
def prepare_sequence(sequence, to_ix):
    new_sequence = autograd.Variable(torch.LongTensor([to_ix[i] for i in sequence]))
    return new_sequence

def prepare_label(label,label_to_ix, cuda=False):
    var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
    return var

def build_token_to_ix(sentences):
    token_to_ix = dict()
    print(len(sentences))
    for sent in sentences:
        for token in sent:
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix

def build_label_to_ix(labels):
    label_to_ix = dict()
    for label in labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)

def load_MR_data():

    print('Generating Data...')

    num_pairs=int(sys.argv[1])
    rel_num=int(sys.argv[2])
    L=universe.InterpretedLanguage(rel_num,num_pairs)
    k=num_pairs*5*rel_num
    
    branching=sys.argv[4]
    complexity=int(sys.argv[3])
    thedata = L.allexamples(branching,complexity=complexity-1)#L.randomexamples(k,branching,complexity=2)
    random.seed(SEED)
    random.shuffle(thedata)
    datasize=len(thedata)
    
    devtest=L.allexamples(branching,complexity=complexity,min_complexity=complexity)
    random.shuffle(devtest)
    datasize=len(devtest)
    #datasize=len(thedata)
    
    p=0.8
    train_data = thedata+devtest[:int(datasize*p)]
    val_data = devtest[int(datasize*p):int(datasize*(p+(1-p)*0.55))]
    test_data = devtest[int(datasize*(p+(1-p)*0.55)):]#L.allexamples(branching,complexity=6,min_complexity=6)
    print("Total data: {}\nTrain: {}\nVal: {}\nTest: {}".format(datasize+len(thedata),len(train_data),len(val_data),len(test_data)))

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    word_to_ix = build_token_to_ix([s for s,_ in train_data+val_data+test_data])
    label_to_ix = {val:idx for idx,val in enumerate(L.names)}
    print("word_to_ix = ", word_to_ix)
    print("label_to_ix = ", label_to_ix)
    print('vocab size:',len(word_to_ix),'label size:',len(label_to_ix))
    print("Data generation done.")

    return train_data,val_data,test_data,word_to_ix,label_to_ix,complexity

