# CHAR RNN

This repository trains/evaluates a char-rnn for text generation.

## Setup

This repository uses Python3 with PyTorch. Run the following command to setup python environment.

```
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Architecture

This network uses a 1-to-1 RNN structure.
For each input character/byte, the network predicts the next character/byte.

## Loss Function

This is a classification problem, therefore I used a softmax cross entropy.
Since the network is trained on a sequence of characters/bytes, it accumulates
all the losses in the character sequence and optimize based on that.


## Train

To train this network, run *train.py* with a directory of text data you want to train.

```
./train.py <checkpoint-name> /path/to/dataset/directory/
```

## Evaluate

To evaluate this network, run *evaluate.py* with a starting mesage and a length for text generation.

```
./evaluate.py --model /path/to/checkpoint.pkl --message <Start Message> --length <num>
```

## Author

[Tianyu Cheng](tianyu.cheng@utexas.edu)
