# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader

__all__ = ("TxtDataset", )


class TxtDataset(Dataset):
    """
    loading bytes from data set
    """

    def __init__(self, chunk_size):
        """ initialize byte dataset with chunk size """
        super(TxtDataset, self).__init__()
        self.bytes = b''
        self.chunk_size = chunk_size


    def add_file(self, filepath: str):
        """ adding a single file into dataset

        Args:
            filepath: path to the text file
        """
        print("--> Add file:", filepath)
        data = bytes(open(filepath, 'rb').read())
        self.bytes += data


    def add_files(self, files: []):
        """ adding a list of files into dataset

        Args:
            files: a list of file paths
        """
        for name in files:
            self.add_file(name)


    def __len__(self):
        """ compute the number of chunks in dataset """
        if len(self.bytes) < self.chunk_size:
            raise RuntimeError("Not enough bytes")
        return len(self.bytes) - self.chunk_size - 2


    def __getitem__(self, idx: int):
        """ get chunk at idx

        Args:
            idx: index into dataset
        """
        X = list(self.bytes[idx + 0:idx + self.chunk_size + 0])
        y = list(self.bytes[idx + 1:idx + self.chunk_size + 1])
        X = torch.LongTensor(X)
        y = torch.LongTensor(y)
        return X, y


    def loader(self, *args, **kwargs):
        """ Returns a pytorch DataLoader object

        Args:
            args, kwargs: arguments from pytorch.utils.data.DataLoader
        """
        return DataLoader(dataset=self, *args, **kwargs)
