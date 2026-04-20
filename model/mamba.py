import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import pandas as pd
from model.utils import *
import logging
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import json


FEATS = {
    'tech': ['RSI', 
             'Mayer_multiple',
             'Volume'],
    'onchain': ['price_vs_ma',
                'price_ma7_ma30',
                'price_ma30_ma90',
                'HashRate_ma7_ma30',
                'HashRate_ma30_ma90',
                'AdrBalCnt_ma7_ma30',
                'AdrBalCnt_ma30_ma90',
                'mvrv_zscore',
                'price_ma7_ma30_gradient',
                'price_ma30_ma90_gradient'],
    'poly': ['btc_sentiment',
             'rate_up_market_count',
             'rate_down_market_count']
}

class CMamba(nn.Module):
    """
    Use Mamba achietecture as basic blocks of prediction model
    """
    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 n_layers: int = 4,
                 d_state: int = 64,
                 d_conv: int = 4,
                 expand: int = 2,
                 dropout: float = 0.1,
                 n_horizons: int = 4,):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        blocks = []

        for _ in range(n_layers):
            blocks.append(
                nn.Sequential(
                    Mamba(
                        d_model = d_model,
                        d_state = d_state,
                        d_conv = d_conv,
                        expand = expand,
                    ),
                    nn.Dropout(dropout),
                    nn.LayerNorm(d_model),
                )
            )
        self.mamba_stack = nn.ModuleList(blocks)

        self.pool = lambda x: x.mean(dim=1)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_horizons),
        )

    def forward(self, x):
        """
        x: (b,l,d)
        reutrns: (b, n_horizons)
        """
        x = self.input_proj(x)
        for block in self.mamba_stack:
            x = x + block(x)

        out = self.head(self.pool(x))

        return out    

def load_data(X: np.ndarray,
              Y: np.ndarray,
              batch_size: int = 128,
              val_ratio: float = 0.2,
              test_ratio: float = 0.1):
    """
    Load features and targets data for training.
    The last 10% time steps (365 at least) is test data. Test data never involved in training and validation.
    The validation data is 20% time steps before test data, and mask 120 last time steps.
    The remaining is training data, which is no more than 70% of total time steps.
    Args:
        X: 16 features in FEATS
        Y: future returns in 4 horizons (30d, 60d, 90d, 120d)
    Returns:
        train_loader, val_loader, test_loader
    """
    
    N = X.shape[0]
    n_test = max(int(N * test_ratio), 365)
    n_val = int(N * val_ratio)
    n_train = N - n_val - n_test

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train: n_train + n_val - 120], Y[n_train: n_train + n_val - 120]
    X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]

    X_scaler = RobustScaler(quantile_range=(2.5, 97.5))
    X_scaler.fit(X_train)
    X_train_scaled = torch.tensor(X_scaler.transform(X_train), dtype=torch.float32)
    X_val_scaled = torch.tensor(X_scaler.transform(X_val), dtype=torch.float32)
    X_test_scaled = torch.tensor(X_scaler.transform(X_test), dtype=torch.float32)

    Y_scalers = [RobustScaler(quantile_range=(2.5, 97.5)),
                RobustScaler(quantile_range=(1, 99)),
                RobustScaler(quantile_range=(1, 99)),
                RobustScaler(quantile_range=(1, 99))]
    assert len(Y_scalers) == Y.shape[1]
    Y_train_scaled = np.zeros_like(Y_train)
    Y_val_scaled = np.zeros_like(Y_val)
    Y_test_scaled = np.zeros_like(Y_test)
    for j in range(len(Y_scalers)):
        Y_train_scaled[:, j] = Y_scalers[j].fit_transform(Y_train[:, j].reshape(-1, 1)).flatten()
        Y_val_scaled[:, j] = Y_scalers[j].transform(Y_val[:, j].reshape(-1, 1)).flatten()
        Y_test_scaled[:, j] = Y_scalers[j].transform(Y_test[:, j].reshape(-1, 1)).flatten()

    def make_batch(X, Y, seq_len=batch_size):
        """
        Use previous 128 steps features to predict today's future return
        """
        T = len(X)
        XX, YY = [], []
        for t in range(seq_len, T):
            XX.append(X[t-seq_len: t])
            YY.append(Y[t])

        XX = np.stack(XX)    
        YY = np.stack(YY)
        dataset = TensorDataset(torch.tensor(XX, dtype=torch.float32), 
                                torch.tensor(YY, dtype=torch.float32))

        return dataset

    train_loader = DataLoader(
        make_batch(X_train_scaled, Y_train_scaled),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        make_batch(X_val_scaled, Y_val_scaled),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        make_batch(X_test_scaled, Y_test_scaled),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

def add_gradient_noise(model, noise_std=0.01):
    """Add noise to gradients for better generalization"""
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_std
            param.grad.add_(noise)

def train(
        model,
        train_loader,
        val_loader,
        n_epochs=50,
        lr=1e-4,
        weight_decay=1e-2,
        use_amp=True
):
    """
    Train loop
    Three things help stablize training:
        1. Adding random noise to gradient.
        2. L2 regularization penalty.
        3. Learning scheduler.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                    max_lr=lr,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=n_epochs,
                                                    pct_start=0.5)

    best_val_loss = float('inf')
    best_state = None
    pbar = tqdm(range(1, n_epochs+1), desc='Training...')

    for idx, e in enumerate(pbar):
        model.train()
        train_loss = 0.0

        for (x, y) in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = model(x)

                loss = F.mse_loss(preds, y)

                # L2 norm penalty
                loss = loss + sum(p.norm(2) for p in model.parameters()) * 0.001
                
                # DEBUG: Check loss 
                if torch.isnan(loss):
                    print(f"NaN loss at epoch {e}, batch {idx}")
                    print(f"Preds sample: {preds[0]}")
                    print(f"Target sample: {y[0]}")
                    print(f"Loss components: {((preds - y)**2).mean()}")
                    return model

            scaler.scale(loss).backward()
            add_gradient_noise(model, noise_std=0.001)
            scaler.step(optimizer)
            scaler.update()

            # lr scheduler
            scheduler.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    preds = model(x)
                    # no L2 norm penalty in validation
                    loss = F.mse_loss(preds, y)
                
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        pbar.set_description(f'Epoch {e:03d} | train: {train_loss: .6f} | val: {val_loss: .6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss

def evaluate(model, data_loader):
    """
    Evaluation loop.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device, non_blocking=True)
            pred = model(x)
            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    mse = ((preds - targets) ** 2).mean(axis=0)
    mae = np.abs((preds - targets) / targets).mean(axis=0)
    sign_acc = (np.sign(preds) == np.sign(targets)).mean(axis=0)

    metrics = {
        'mse': mse.tolist(),
        'mae': mae.tolist(),
        'sign_acc': sign_acc.tolist()
    }

    return metrics, preds, targets

def predict(model, x: torch.Tensor):
    """
    One time step prediction
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        x = x.unsqueeze(0).to(device)
        preds = model(x).cpu().numpy()[0]

    return preds

def prepare_data(start='2010-07-18', end='2025-12-31'):
    """
    prepare dataset from data source.
    The 3 functions to get data are imported from utils.py
    Returns:
        X: np.ndarray N * 16 features
        Y: np.ndarray N * 4 horizons
    """
    tech = compute_technical_metrics(window=14)[FEATS['tech']]
    onchain = compute_onchain_features()[FEATS['onchain']]
    poly, _, _ = compute_polymarket_features()
    poly = poly[FEATS['poly']]

    X = pd.concat([tech, onchain, poly], 
                  axis=1, 
                  sort=True).loc[start: end].sort_index().fillna(0)

    Y = compute_btc_returns()[['return_030d', 
                               'return_060d', 
                               'return_090d', 
                               'return_120d']].loc[start: end].sort_index().values

    return X.values, Y

def init_weights(model):
    """
    Safe initialization that won't crash
    Bad initialization may cause Nan problems. Xavier is fine.
    """
    
    for name, param in model.named_parameters():
        # Check if tensor has at least 2 dimensions for Xavier
        if param.dim() >= 2 and 'weight' in name:
            if 'mamba' in name.lower():
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'head' in name.lower() or 'output' in name.lower():
                nn.init.xavier_uniform_(param, gain=0.1)
            else:
                nn.init.xavier_uniform_(param, gain=1.0)
        
        # Handle biases
        elif 'bias' in name:
            nn.init.constant_(param, 0.01)
            if 'forget' in name or 'gate' in name:
                nn.init.constant_(param, 1.0)
        
        # Handle LayerNorm weights (1D)
        elif 'weight' in name and 'norm' in name:
            nn.init.constant_(param, 1.0)
        
        # Default for anything else
        elif param.dim() == 1:
            nn.init.constant_(param, 0.0)

def plot_eval(preds, y, end_date, col=['30d','60d','90d','120d']):
    """
    plot predictions Vs ground truth
    """
    fig, ax = plt.subplots(2,2)
    coordinates = [(i, j) for j in range(2) for i in range(2)]
    k = 0
    for l in range(len(col)):
        i, j = coordinates[k]
        sd = np.array(preds[:,l]-y[:,l]).std()
        ax[i][j].plot(np.arange(preds.shape[0]), preds[:,l], label='preds')
        ax[i][j].fill_between(np.arange(preds.shape[0]), preds[:,l]-sd, preds[:,l]+sd, alpha=0.2)
        ax[i][j].plot(np.arange(preds.shape[0]), y[:,l], label='true')
        ax[i][j].set_title(f'Preds Vs True - {col[l]}')
        ax[i][j].legend()
        k += 1
    # plt.show()
    plt.savefig(f'model/checkpoint/model_{end_date}_eval_plot.png')
    plt.close()

def run(start='2010-07-18',end='2025-12-31'):
    """
    A single run of training and evaluation
    """

    print(f'loading data from {start} to {end}...')
    X, Y = prepare_data(start, end)

    train_loader, val_loader, test_loader = load_data(X, Y, batch_size=128)

    model = CMamba(
        input_dim=X.shape[1],
        d_model=256,      
        n_layers=4,
        d_state=64,
        d_conv=4,
        expand=2,
        dropout=0.1,
        n_horizons=4,
    )
    model.apply(init_weights)

    # print('training...')
    best_model, best_val_loss = None, float('inf')
    best_preds, y = None, None
    for retry in range(3):
        print(f'trying {retry}/3')
        model, val_loss = train(model, train_loader, val_loader, n_epochs=150  , lr=1e-3)     # type: ignore
        metrics, preds, targets = evaluate(model, test_loader)
        if val_loss < best_val_loss:
            best_model = model
            best_metrics = metrics
            best_preds = preds
            y = targets
            best_val_loss = val_loss
        
    torch.save(best_model.state_dict(), f'model/checkpoint/model_{end}.pt') # type: ignore
    with open(f'model/checkpoint/model_{end}_metrics.json', 'w') as file:
        json.dump(best_metrics, file) # type: ignore

    print("MSE per horizon [30,60,90,120]:", best_metrics["mse"]) # type: ignore
    print("MAE_pct per horizon [30,60,90,120]:", best_metrics["mae"]) # type: ignore
    print("Directional accuracy per horizon[30,60,90,120]:", best_metrics["sign_acc"]) # type: ignore

    plot_eval(preds=best_preds, y=y, end_date=end)

def main():
    """
    Train the model every 6 months. The models are saved in model/checkpoint and named by last day of dataset.
    Refer to desc of function load_data (line 97):
        1. The last 365 time steps are test data, not involved in training and validation.
        2. The last 120 steps of validation data have been masked to prevent info leakage.
        3. In mamba_backtest.py, select corresponding model to predict signals.
    """
    END_DATE = ['2018-12-31', '2019-06-30', '2019-12-31', '2020-06-30', '2020-12-31', 
                '2021-06-30', '2021-12-31', '2022-06-30', '2022-12-31', '2023-06-30', '2023-12-31',
                '2024-06-30', '2024-12-31', '2025-06-30', '2025-12-31']
    for end in END_DATE:
        run(start='2010-07-18', end=end)

if __name__ == '__main__':
    main()