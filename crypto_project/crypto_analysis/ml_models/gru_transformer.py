import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {'GPU' if torch.cuda.is_available() else 'CPU'}")


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


class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUAttentionModel, self).__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_scores = self.attention(gru_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        out = self.fc(context)
        return out


def train_gru_attention(
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
    validation_data=None,
    early_stopping_rounds=None,
):
    X_train = impute_missing(X_train, method=impute_method)
    y_train = impute_missing(y_train, method=impute_method)
    X_seq, y_seq = create_sequences(X_train, y_train, seq_len)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_seq.reshape(-1, 1), dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = X_train.shape[1]
    model = GRUAttentionModel(input_size, hidden_size, num_layers, dropout).to(device)
    print(f"Модель GRUAttentionModel перемещена на {next(model.parameters()).device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"GRU_Attention Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
        )

        if validation_data is not None and early_stopping_rounds is not None:
            X_val, y_val = validation_data
            X_val = impute_missing(X_val, method=impute_method)
            y_val = impute_missing(y_val, method=impute_method)
            X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
            X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(
                y_val_seq.reshape(-1, 1), dtype=torch.float32
            ).to(device)
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            model.train()
            print(f"  Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state_dict = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_rounds:
                    print(f"  Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_state_dict)
                    return model
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
    validation_data=None,
    early_stopping_rounds=None,
):
    X_train = impute_missing(X_train, method=impute_method)
    y_train = impute_missing(y_train, method=impute_method)
    X_seq, y_seq = create_sequences(X_train, y_train, seq_len)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_seq.reshape(-1, 1), dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_size = X_train.shape[1]
    model = TransformerModel(input_size, d_model, num_layers, nhead, dropout).to(device)
    print(f"Модель TransformerModel перемещена на {next(model.parameters()).device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
            f"Transformer Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
        )

        if validation_data is not None and early_stopping_rounds is not None:
            X_val, y_val = validation_data
            X_val = impute_missing(X_val, method=impute_method)
            y_val = impute_missing(y_val, method=impute_method)
            X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
            X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(
                y_val_seq.reshape(-1, 1), dtype=torch.float32
            ).to(device)
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            model.train()
            print(f"  Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state_dict = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_rounds:
                    print(f"  Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_state_dict)
                    return model
    return model
