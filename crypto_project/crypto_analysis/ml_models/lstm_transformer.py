import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


def impute_missing(df, method="ffill"):
    if method == "ffill":
        return df.ffill().bfill()
    else:
        raise ValueError(f"Unsupported imputation method: {method}")


def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X.iloc[i : i + seq_len].values)
        y_seq.append(y.iloc[i + seq_len])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def train_lstm(
    X_train,
    y_train,
    seq_len,
    epochs,
    batch_size,
    lr,
    dropout,
    impute_method,
    hidden_size,
    num_layers,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = impute_missing(X_train, method=impute_method)
    y_train = impute_missing(y_train, method=impute_method)

    X_seq, y_seq = create_sequences(X_train, y_train, seq_len)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_seq.reshape(-1, 1), dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = X_train.shape[1]
    model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"LSTM Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
        )
    return model


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, num_layers, nhead, dropout):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


def train_transformer(
    X_train,
    y_train,
    seq_len,
    epochs,
    batch_size,
    lr,
    dropout,
    impute_method,
    d_model,
    num_layers,
    nhead,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = impute_missing(X_train, method=impute_method)
    y_train = impute_missing(y_train, method=impute_method)

    X_seq, y_seq = create_sequences(X_train, y_train, seq_len)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_seq.reshape(-1, 1), dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = X_train.shape[1]
    model = TransformerModel(input_size, d_model, num_layers, nhead, dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Transformer Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
        )
    return model
