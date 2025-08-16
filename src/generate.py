import torch
from .model import LSTMModel
from .data import load_songs, vectorize_string, map_characters_and_numbers
from tqdm import tqdm
import mitdeeplearning as mdl
from IPython import display
import numpy as np
from scipy.io.wavfile import write

#install abcMIDI timidy > /dev/null 2>&1

MODEL_PATH = "./training_checkpoints/ckpt"
_, VOCAB = load_songs()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def load_model() -> LSTMModel:
    params = dict(
        num_training_iterations = 3000,
        batch_size = 8,
        seq_length = 100,
        learning_rate = 5e-3,
        embedding_dim = 256,
        hidden_size = 1024
    )

    model = LSTMModel(
        len(VOCAB),
        params['embedding_dim'],
        params['hidden_size']
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    return model

def generate_text(model: LSTMModel, start_string: str, generation_length=1000) -> str:
    char2idx, idx2char = map_characters_and_numbers(VOCAB)
    input_idx = vectorize_string(start_string, char2idx)

    input_idx = torch.tensor([input_idx], dtype=torch.long).to(DEVICE)

    state = model.init__hidden(input_idx.size(0), DEVICE)
    text_generated = []

    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        predictions, state = model(input_idx, state, return_state=True)

        predictions = predictions.squeeze(0)

        input_idx = torch.multinomial(torch.softmax(predictions, dim=-1), num_samples=1)

        text_generated.append(idx2char[input_idx].item())

    return start_string + "".join(text_generated)


def main():
    model = load_model()

    starting_string = input("Starting screen: ")

    generated_text = generate_text(model, starting_string)
    print(generated_text)

    generated_songs = mdl.lab1.extract_song_snippet(generated_text)

    for i, song in enumerate(generated_songs):
        waveform = mdl.lab1.play_song(song)

        if waveform:
            display.display(waveform)

            numeric_data = np.frombuffer(waveform.data, dtype=np.int16)

            wav_file_path = f"output_{i}.wav"
            write(wav_file_path, 88200, numeric_data)


if __name__ == '__main__':
    main()