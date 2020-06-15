import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

"""
Blog post:
Taming LSTMs: Variable-sized mini-batches and why PyTorch is good for your health:
https://medium.com/@_willfalcon/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
"""


class myLSTM(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, num_layers):
        super(myLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)))

        
    
    def forward(self, X):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        embeddings = self.word_embedding(X)
        x = embeddings.view(len(X), 1, -1)

        # now run through LSTM
        y, self.hidden = self.lstm(x, self.hidden)
        out = self.hidden2label(y[-1])
        return out

        # undo the packing operation
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, hidden_dim) -> (batch_size * seq_len, hidden_dim)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        # X = X.contiguous()
        # X = X.view(-1, X.shape[2])

        # run through actual linear layer
        # X = self.hidden_to_tag(X)
        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, hidden_dim) -> (batch_size, seq_len, nb_tags)
        # X = F.log_softmax(out, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        # X = X.view(batch_size, seq_len, self.nb_tags)


    
    def loss(self, y_hat, y):
        criterion = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=0)
        ce_loss = criterion(y_hat, y.argmax())
        return ce_loss