import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class LoremRNN(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(LoremRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))

    def get_meta(self):
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "embedding_size": self.embedding_size,
            "n_layers": self.n_layers
        }


def index(text):
    char_counts = Counter(text)
    char_counts = sorted(char_counts.items(), key = lambda x: x[1], reverse=True)

    sorted_chars = [char for char, _ in char_counts]
    char_index = {char: index for index, char in enumerate(sorted_chars)}
    index_char = {v: k for k, v in char_index.items()}
    seq = np.array([char_index[char] for char in text])

    return seq, index_char, char_index


def init_rnn(file_name, hidden_size, embedding_size, n_layers):

    with open(file_name, 'r', encoding='utf-8') as data_file:
        data = " ".join(data_file.readlines())
    seq, index_char, char_index = index(data)

    model = LoremRNN(input_size=len(index_char), hidden_size=hidden_size,
                     embedding_size=embedding_size, n_layers=n_layers)
    model.to(device)
    return model, seq, index_char, char_index


def evaluate(model, char_to_idx, idx_to_char, start_text=' ', prediction_len=200, temp=0.3):
    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    _, hidden = model(train, hidden)

    inp = train[-1].view(-1, 1, 1)

    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char

    return predicted_text
