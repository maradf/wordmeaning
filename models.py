import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

"""
Blog post:
Taming LSTMs: Variable-sized mini-batches and why PyTorch is good for your health:
https://medium.com/@_willfalcon/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
"""
""" 
class myLSTM(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, num_targets):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_targets = num_targets
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.num_targets)
        
        self.h = torch.zeros((2, batch_size, hidden_dim))
        self.c = torch.zeros((2, batch_size, hidden_dim))
        

        
    def forward(self, x):

        x = self.embeddings(x)
        self.batch_size = x.size(0)
        
        seq_len = x.size(1)
        seq_len = torch.as_tensor([seq_len], dtype=torch.int64, device='cpu')
    
        #x = self.dropout(x)
#        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(x) # x_pack
    


        #out = out_pack
        out = self.linear(out)
        return out

    def load(self, location):
        self.lstm.load_state_dict(torch.load(location))
        self.lstm.eval()

"""
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
        return (autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)))

        
    
    def forward(self, X):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        embeddings = self.word_embedding(X)
        x = embeddings.view(len(X), 1, -1)

        # now run through LSTM
        y, self.hidden = self.lstm(x, self.hidden)
        print("y = ", y.shape)
        out = self.hidden2label(y[-1])
        print("out size = ", out.size())
        print("hidden_to_label", self.hidden2label)
        return out


    
    def loss(self, y_hat, y):
        criterion = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=0)
        ce_loss = criterion(y_hat, y.argmax())
        return ce_loss
