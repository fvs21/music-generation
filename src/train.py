import torch
import os
from .data import *
from .model import LSTMModel
from typing import Any
import mitdeeplearning as mdl
from tqdm import tqdm

cross_entropy = torch.nn.CrossEntropyLoss()

def compute_loss(labels: Tensor, logits: Tensor) -> Any:
    batched_labels = labels.view(-1)

    batched_logits = logits.view(-1, logits.size(-1))

    loss = cross_entropy(batched_logits, batched_labels)

    return loss

def train_step(model: LSTMModel, x: Tensor, y: Tensor, optimizer: torch.optim.Adam) -> Any:
    model.train()

    optimizer.zero_grad()

    y_hat = model(x)

    loss = compute_loss(y, y_hat)

    loss.backward()
    optimizer.step()

    return loss


def main():
    songs, vocab = load_songs()
    char2idx, idx2char = map_characters_and_numbers(vocab)
    vectorized_songs = vectorize_string(songs, char_mapping=char2idx)

    vocab_size = len(vocab)

    params = dict(
        num_training_iterations = 3000,
        batch_size = 8,
        seq_length = 100,
        learning_rate = 5e-3,
        embedding_dim = 256,
        hidden_size = 1024
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(vocab_size, params["embedding_dim"], params["hidden_size"]).to(device)

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

    example_batch_loss = compute_loss(y, pred)
    print(f"Prediction shape: {pred.shape} # (batch_size, sequence_length, vocab_size)")
    print(f"scalar_loss:      {example_batch_loss.mean().item()}")


    #start training
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    os.makedirs(checkpoint_dir, exist_ok=True)

    history = []
    plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    if hasattr(tqdm, '_instances'): tqdm._instances.clear()

    for iter in tqdm(range(params["num_training_iterations"])):
        x_batch, y_batch = get_batch(vectorized_songs, params["seq_length"], params["batch_size"])

        x_batch = torch.tensor(x_batch, dtype=torch.long).to(device)
        y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)

        loss = train_step(model, x_batch, y_batch, optimizer=optimizer)

        history.append(loss.item())
        plotter.plot(history)

        if iter % 100 == 0:
            torch.save(model.state_dict(), checkpoint_prefix)

    torch.save(model.state_dict(), checkpoint_prefix)
    print("Finished training!")



if __name__=='__main__':
    main()