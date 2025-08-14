import torch
import numpy as np
import os
import time
import functools
from scipy.io.wavfile import write
from .data import *
from .model import LSTMModel

def main():
    songs, vocab = load_songs()
    char2idx, idx2char = map_characters_and_numbers(vocab)
    vectorized_songs = vectorize_string(songs, char_mapping=char2idx)

    vocab_size = len(vocab)
    embedding_dim = 256
    hidden_size = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(vocab_size, embedding_dim, hidden_size).to(device)

    print(model)

    x, y = get_batch(vectorized_songs=vectorized_songs, sequence_len=100, batch_size=32)

    x = x.to(device)
    y = y.to(device)

    pred = model(x)

    sampled = torch.multinomial(torch.softmax(pred[0], dim=-1), num_samples=1)
    sampled = sampled.squeeze(-1).cpu().numpy()
    
    #Before training
    print("Input: \n", repr("".join(idx2char[x[0].cpu()])))
    print("Next Char Predictions: \n", repr("".join(idx2char[sampled])))


def compute_loss():
    raise NotImplementedError

def train_step():
    raise NotImplementedError


if __name__=='__main__':
    main()