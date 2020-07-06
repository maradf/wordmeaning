"""
Written by: Dr. Denis Paperno (Paperno (2018))
Edited and documented by: Mara Fennema

Class definition of an LSTM with the parameters to result in the optimal
LSTM usable by train.py.
"""
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

class myLSTM(nn.Module):
    """ myLSTM class to create LSTM with specific features, as required by the
        code in train.py. 

    Args:
        embedding_dim (int): Number of dimensions required by the embedding.
        hidden_dim (int): Number of dimensions for the hidden layerof the LSTM.
        vocab_size (int): Number of unique characters possible in the input.
        label_size (int): Number of unique labels possible as the output.
        num_layers (int): Number of hidden layers for the LSTM.
        bidirectional (bool): Boolean denoting whether the should be 
                                bidirectional (True) or not (False).
    """    


    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, num_layers, bidirectional):
        super(myLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions=1+int(bidirectional)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=num_layers,
                            bidirectional=bidirectional)
        self.hidden2label = nn.Linear(hidden_dim * self.num_directions,
                                      label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """ Creates the hidden state consisting of a hidden and a cell. 
        
        Returns:
            tuple: Tuple containing the hidden and the cell of the LSTM.
        """
        return (autograd.Variable(torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_dim)))

        
    
    def forward(self, X):
        """ Gets output of the LSTM when using the given input.

        Args:
            X (torch.LongTensor): Input tensor of the LSTM, based on which the
                                  LSTM needs to predict the output.
        
        Returns:
            out (torch.Tensor): Output tensor containing the predictions for 
                                each possible label. 
        """
        # Resets the LSTM hidden state. Must be done before a new batch is run,
        # otherwise the LSTM will treat a new batch as a continuation 
        # of a sequence.
        embeddings = self.word_embedding(X)
        x = embeddings.view(len(X), 1, -1)

        # Run through the LSTM
        y, self.hidden = self.lstm(x, self.hidden)
        
        # Perform linear transformation to allow the output shape to be the 
        # length of all possible labels multiplied by the number of directions.
        out = self.hidden2label(y[-1])
        return out
