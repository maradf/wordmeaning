import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

class myLSTM(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, num_layers, bidirectional):
        super(myLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions=1+int(bidirectional)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional)
        self.hidden2label = nn.Linear(hidden_dim*self.num_directions, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_dim)))

        
    
    def forward(self, X):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        embeddings = self.word_embedding(X)
        x = embeddings.view(len(X), 1, -1)

        # now run through LSTM
        y, self.hidden = self.lstm(x, self.hidden)
        out = self.hidden2label(y[-1])
        return out


    
    def loss(self, y_hat, y):
        criterion = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=0)
        ce_loss = criterion(y_hat, y.argmax())
        return ce_loss
