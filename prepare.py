import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn

import RNN

# static parameters
SEQ_LEN = 256
BATCH_SIZE = 16
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def seq_to_batch(seq):
    """
    generate batches from sequence
    """
    trains = []
    targets = []
    for k in range(BATCH_SIZE):
        batch_start = np.random.randint(0, len(seq) - SEQ_LEN)
        chunk = seq[batch_start: batch_start + SEQ_LEN]
        train = torch.LongTensor(chunk[:-1]).view(-1, 1)
        target = torch.LongTensor(chunk[1:]).view(-1, 1)
        trains.append(train)
        targets.append(target)
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)


if __name__ == '__main__':
    # model path parameter
    model_path = sys.argv[2]
    data = ""
    model, seq, index_char, char_index = RNN.init_rnn(sys.argv[1], hidden_size=128, embedding_size=128, n_layers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        verbose=True,
        factor=0.5
    )

    n_epochs = 50000
    loss_avg = []
    if os.path.exists(model_path):
        print("Load model...")
        model.load_state_dict(torch.load(model_path))

    try:
        for epoch in range(n_epochs):
            model.train()
            train, target = seq_to_batch(seq)
            train = train.permute(1, 0, 2).to(device)
            target = target.permute(1, 0, 2).to(device)
            hidden = model.init_hidden(BATCH_SIZE)

            output, hidden = model(train, hidden)
            loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_avg.append(loss.item())
            if len(loss_avg) >= 50:
                mean_loss = np.mean(loss_avg)
                print(f'Loss: {mean_loss}')
                scheduler.step(mean_loss)
                loss_avg = []
                model.eval()
                predicted_text = RNN.evaluate(model, char_index, index_char)
                print(predicted_text)
    except KeyboardInterrupt:
        print("Canceled by user")

    print("Save model...")
    torch.save(model.state_dict(), model_path)
    with open('meta', 'w') as meta_data:
        json.dump({
            "model": model.get_meta(),
            "file_name": sys.argv[1]
        }, meta_data)
