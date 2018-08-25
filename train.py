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
    parser = argparse.ArgumentParser(prog="CharRNN", description="CharRNN training")
    # input parameters
    parser.add_argument("model_name",       type=str,   help="model output name")
    parser.add_argument("data_directory",   type=str,   help="data directory for training")
    # network parameters
    parser = parameters.parse_network_parameters(parser)
    # training parameters
    parser = parameters.parse_training_parameters(parser)
    return parser.parse_args()


class Trainer(object):
    """ char rnn trainer """

    def __init__(self, args, debug=False):
        super(Trainer, self).__init__()
        self.device = self._init_device_(debug)
        self.network = model.CharRNN.create(args).to(self.device)
        self.current_epoch = 0
        self.current_step = 0


    def _init_device_(self, debug):
        """ initialize cpu/gpu device """
        if debug:
            device = torch.device("cpu")
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return device


    def loader(self, args):
        """ create dataset loader  """
        dataset = reader.TxtDataset(parameters.chunk_size)
        for root, dirs, names in os.walk(args.data_directory):
            for name in names:
                dataset.add_file(os.path.join(root, name))
        return dataset.loader(batch_size=args.batch_size)


    def save(self, args, epoch, step):
        """ save network parameters """
        directory = Path("checkpoints") / args.model_name
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / "{}-{}.pkl".format(epoch, step)
        torch.save(self.network.state_dict(), str(filename))
        print("checkpoint saved:", filename)


    def load(self, args):
        """ load latest trained model """
        directory = Path("checkpoints") / args.model_name
        if not directory.exists():
            return
        files = [f[:-len(".pkl")].split("-") for f in os.listdir(str(directory))]
        if files:
            files = sorted(files, key=lambda x: float("{}.{}".format(x[0], x[1])), reverse=True)
            self.current_epoch = int(files[0][0])
            self.current_step = int(files[0][1])
            filename = str(directory / "{}-{}.pkl".format(self.current_epoch, self.current_step))
            states = torch.load(filename)
            self.network.load_state_dict(states)
            print("checkpoint loaded:", filename)


    def train(self, args):
        """ training """
        loader    = self.loader(args)
        network   = self.network
        optimizer = torch.optim.Adam(network.parameters(), args.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        network.train(mode=True)
        self.load(args)
        # return

        print("NET  |", network,   "\n" + "-" * 80)
        print("OPT  |", optimizer, "\n" + "-" * 80)
        print("LOSS |", criterion, "\n" + "-" * 80)

        def optimize(loss):
            """ one optimize step """
            network.zero_grad()
            loss.backward()
            optimizer.step()

        def step(batch_x, batch_y):
            """ one training step """
            loss = 0
            hidden = self.network.init_hidden(batch_x.size(0))
            hidden = tuple([h.to(self.device) for h in hidden])
            for i in range(args.chunk_size):
                output, hidden = self.network(batch_x[:, i], hidden)
                net_loss = criterion(output, batch_y[:, i])
                loss += net_loss
            return loss / args.chunk_size

        def report(args, epoch, step, loss):
            if step % 10 == 0:
                print("epoch: {:<8} step: {:<8} loss: {:<8}".format(epoch, step, loss))
            if step % 100 == 0 and step != 0:
                self.save(args, epoch, step)

        try:
            for epoch in range(self.current_epoch, args.num_epochs):
                self.current_epoch = epoch
                for i, (batch_x, batch_y) in enumerate(loader):
                    self.current_step += 1
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    loss = step(batch_x, batch_y)
                    report(args, self.current_epoch, self.current_step, loss)
                    optimize(loss)
        except (EOFError, KeyboardInterrupt):
            self.save(args, epoch, i)
            sys.exit(0)



if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args, debug=False)
    trainer.train(args)
