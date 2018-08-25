# -*- coding: utf-8 -*-

import torch
import numpy as np
import parameters

__all__ = ("CharRNN", )


class CharRNN(torch.nn.Module):

    """ Char RNN """

    def __init__(self, vocabulary_size, embedding_size, hidden_size, num_layers, rnn_type="LSTM"):
        """ initialize RNN structure

        Args:
            vocabulary_size:    number of distinct characters in vocabulary (for arbitrary bytes input, use 256)
            embedding_size:     character/word embedding vector dimension
            hidden_size:        RNN hidden units
            num_layers:         number of layers in RNN
            rnn_type:           type of RNN, e.g. LSTM, GRU, etc
        """
        super(CharRNN, self).__init__()
        # record fields
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        # network structure
        # --> 1. learning word/character embedding (vocabulary_size -> embedding_size)
        self.encoder = torch.nn.Embedding(vocabulary_size, embedding_size)

        # --> 2. RNN for character prediction
        if self.rnn_type == "LSTM":
            self.rnn = torch.nn.LSTM(input_size=embedding_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers)
        else:
            raise RuntimeError("Unsupported RNN type: {}".format(self.rnn_type))

        # --> 3. fully connected layer to learn transformation from embedding space back
        self.decoder = torch.nn.Linear(hidden_size, vocabulary_size)


    def init_hidden(self, batch_size):
        """ init size for RNN module

        Args:
            batch_size: number of items in a batch
        """
        if self.rnn_type == "LSTM":
            h_hx = torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
            h_cx = torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
            return h_hx, h_cx
        else:
            raise RuntimeError("Unsupported RNN type: {}".format(self.rnn_type))


    def forward(self, X, hidden):
        """ compute forward step for network

        Args:
            X:      network input
            hidden: hidden state
        """
        batch_size = X.size(0)
        # --> 1. encode
        encoded = self.encoder(X)
        # --> 2. rnn
        encoded = encoded.view(1, batch_size, self.embedding_size)
        output, hidden = self.rnn(encoded, hidden)
        # --> 3. decode
        output = output.view(batch_size, -1)
        output = self.decoder(output)
        return output, hidden


    @staticmethod
    def create(args):
        return CharRNN(vocabulary_size=parameters.vocabulary_size,
                       embedding_size=args.embedding_size,
                       hidden_size=args.hidden_size,
                       num_layers=args.num_layers,
                       rnn_type=args.rnn_type)
