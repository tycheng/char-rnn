# -*- coding: utf-8 -*-

import torch

# for arbitrary byte input, we choose vocabulary
# to be simply a byte (256 characters)
vocabulary = [ chr(i) for i in range(256) ]
vocabulary_size = len(vocabulary)
embedding_size = vocabulary_size // 16

# training hyper-parameters
learning_rate = 0.001
num_epochs = 1000
batch_size = 512
chunk_size = 128

# RNN parameters
num_layers = 1
hidden_size = 256
rnn_type = "LSTM"


def parse_network_parameters(parser):
    # network parameters
    parser.add_argument("--embedding_size", type=int,   help="vocabulary embedding size",     default=embedding_size)
    parser.add_argument("--num-layers",     type=int,   help="training RNN layers",           default=num_layers)
    parser.add_argument("--hidden-size",    type=int,   help="training RNN hidden unit size", default=hidden_size)
    parser.add_argument("--rnn-type",       type=str,   help="RNN type",                      default=rnn_type)
    return parser

def parse_training_parameters(parser):
    # traning parameters
    parser.add_argument("--learning-rate",  type=float, help="training learning rate",        default=learning_rate)
    parser.add_argument("--num-epochs",     type=int,   help="training learning rate",        default=num_epochs)
    parser.add_argument("--batch-size",     type=int,   help="training batch size",           default=batch_size)
    parser.add_argument("--chunk-size",     type=int,   help="training chunk size",           default=chunk_size)
    return parser
