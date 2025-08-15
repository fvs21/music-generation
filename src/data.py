import mitdeeplearning as mdl
from typing import Tuple, List, Dict
from numpy import ndarray, array, random
from torch import Tensor, tensor, long

def load_songs() -> Tuple[str, List[str]]:
    songs = mdl.lab1.load_training_data()

    songs_joined = "\n\n".join(songs)

    vocab = sorted(set(songs_joined))

    return songs_joined, vocab

def map_characters_and_numbers(vocab: List[str]) -> Tuple[Dict, ndarray]:
    char2idx = {c: i for i, c in enumerate(vocab)}

    idx2char = array(vocab)

    return (char2idx, idx2char)

def vectorize_string(string: str, char_mapping: Dict) -> ndarray:
    vectorized = array([char_mapping[char] for char in string])

    return vectorized

def get_batch(vectorized_songs: ndarray, sequence_len: int, batch_size: int) -> Tuple[Tensor, Tensor]:
    n = vectorized_songs.shape[0] - 1

    idx = random.choice(n - sequence_len, batch_size)

    input_batch = [vectorized_songs[i:i+sequence_len] for i in idx]
    output_batch = [vectorized_songs[i+1:i+sequence_len+1] for i in idx]

    x_batch = tensor(input_batch, dtype=long)
    y_batch = tensor(output_batch, dtype=long)

    return x_batch, y_batch