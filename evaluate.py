#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import numpy as np

import reader
import model
import parameters

from pathlib import Path


def parse_args():
    """ parse parameters """
    # input parameters
    parser = argparse.ArgumentParser(prog="CharRNN Evaluation", description="CharRNN evaluation")
    parser.add_argument("--model",       type=str,   help="pytorch model pickle file")
    parser.add_argument("--message",     type=str,   help="starting message")
    parser.add_argument("--length",      type=int,   help="length for byte generation")
    parser.add_argument("--encoding",    type=str,   default="utf-8", help="byte encoding scheme")
    parser.add_argument("--temperature", type=float, default=0.8, help="randomness for text generation")
    # network parameters
    parser = parameters.parse_network_parameters(parser)
    return parser.parse_args()


class Evaluation(object):
    """ char rnn evaluator """

    def __init__(self, args, debug=False):
        super(Evaluation, self).__init__()
        self.device = self._init_device_(debug)
        self.network = model.CharRNN.create(args).to(self.device)
        self.load(args)


    def _init_device_(self, debug):
        """ initialize cpu/gpu device """
        if debug:
            device = torch.device("cpu")
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return device


    def load(self, args):
        """ load specified model """
        states = torch.load(args.model)
        self.network.load_state_dict(states)
        print("checkpoint loaded:", args.model)


    def run(self, args):
        """ evaulation """
        message = list(bytes(args.message, args.encoding))

        # --> batch size is 1
        inp = torch.LongTensor(message)
        inp = inp.view(1, -1)

        # --> initalize hidden state
        hidden = self.network.init_hidden(1)
        for i in range(len(message)):
            output, hidden = self.network(inp[:, i], hidden)

        # --> prediction
        inp = inp[:, len(message) - 1]
        for i in range(args.length):
            output, hidden = self.network(inp, hidden)
            distribution = output.data.view(-1).div(args.temperature).exp()
            selected = torch.multinomial(distribution, 1)[0].numpy()
            message.append(int(selected))
            inp = torch.LongTensor([ int(selected) ]).to(self.device).view(1, -1)

        # --> display
        message = bytes(message)
        print(message.decode(encoding="utf-8", errors="ignore"))


if __name__ == "__main__":
    args = parse_args()
    evaluator = Evaluation(args, debug=True)
    evaluator.run(args)
