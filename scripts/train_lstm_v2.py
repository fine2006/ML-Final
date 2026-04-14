"""
DECISION LOG:
- Bidirectional LSTM with Multi-Head Self-Attention
- Separate models for t+1, t+6, t+24 horizons
- CPU-only training (no GPU)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class TimeSeriesDataset(Dataset):
    """Dataset for sequential time series data."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 24):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx : idx + self.seq_len]),
            torch.FloatTensor([self.y[idx + self.seq_len]]),
        )


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-style position awareness."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, (
            "hidden_size must be divisible by num_heads"
        )

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )

        output = self.out_proj(context)

        return output


class BidirectionalLSTMWithAttention(nn.Module):
    """
    Bidirectional LSTM with Multi-Head Self-Attention.

    Architecture:
    1. Input projection
    2. Positional encoding
    3. 2-layer Bidirectional LSTM
    4. Multi-head self-attention
    5. Fully connected output
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        recurrent_dropout: float = 0.3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size)

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention
        self.attention = MultiHeadSelfAttention(
            hidden_size * 2,  # Bidirectional = 2x hidden
            num_heads=num_heads,
            dropout=dropout,
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_out = self.attention(lstm_out)

        # Take last timestep output
        out = self.fc(attn_out[:, -1, :])

        return out


def train_lstm_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 10,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict]:
    """Train LSTM model with early stopping."""

    model = BidirectionalLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        print(
            f"  Epoch {epoch + 1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_state)

    return model, history


def evaluate_lstm(
    model: nn.Module, data_loader: DataLoader, device: str = "cpu"
) -> Dict:
    """Evaluate LSTM model."""
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
        "predictions": predictions,
        "actuals": actuals,
    }


def run_lstm_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seq_len: int = 24,
    horizon: str = "t+1",
) -> Dict:
    """Run complete LSTM experiment."""

    print(f"\n[Training LSTM for {horizon}]")

    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len)
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_len)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(
        f"  Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )

    # Determine input size from first feature dimension
    input_size = X_train.shape[1]

    # Train model
    model, history = train_lstm_model(
        train_loader,
        val_loader,
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        epochs=50,
        lr=0.001,
        patience=10,
        device="cpu",
    )

    # Evaluate
    test_results = evaluate_lstm(model, test_loader, device="cpu")
    print(
        f"  Test RMSE: {test_results['RMSE']:.4f}, MAE: {test_results['MAE']:.4f}, R2: {test_results['R2']:.4f}"
    )

    # Save loss history to JSON for visualizations
    import json

    history_path = f"models/loss_history_{horizon}.json"
    with open(history_path, "w") as f:
        json.dump(history, f)

    # Save predictions for visualization
    np.save(f"models/lstm_predictions_{horizon}.npy", test_results["predictions"])
    np.save(f"models/lstm_actuals_{horizon}.npy", test_results["actuals"])

    return {"model": model, "history": history, "test_results": test_results}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from scripts.load_data_v2 import load_all_regions
    from scripts.preprocess_v2 import (
        preprocess_hourly,
        create_horizon_targets,
        chronological_split,
        scale_features,
        prepare_xy,
    )

    print("Loading data...")
    df = load_all_regions([2022, 2023, 2024, 2025])

    print("Preprocessing for t+1...")
    df_proc = preprocess_hourly(df)
    df_t1 = create_horizon_targets(df_proc, 1)
    train, val, test = chronological_split(df_t1)

    train_scaled, val_scaled, test_scaled, _ = scale_features(train, val, test)
    X_train, y_train = prepare_xy(train_scaled)
    X_val, y_val = prepare_xy(val_scaled)
    X_test, y_test = prepare_xy(test_scaled)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Run LSTM
    results = run_lstm_experiment(
        X_train, y_train, X_val, y_val, X_test, y_test, seq_len=24, horizon="t+1"
    )
    print(f"\nFinal Results: {results['test_results']}")
