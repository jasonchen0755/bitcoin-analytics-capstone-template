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


class CMambaFiveHorizons(nn.Module):
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
    
    N = X.shape[0]
    n_test = int(N * test_ratio)
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


def train(
        model,
        train_loader,
        val_loader,
        n_epochs=50,
        lr=1e-4,
        weight_decay=1e-3,
        use_amp=False
):
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
    patience, patience_counter = 20, 0

    best_val_loss = float('inf')
    best_state = None
    loss_history = []

    for e in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {e}/{n_epochs}')

        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device, non_blocking=False)
            y = y.to(device, non_blocking=False)

            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # DEBUG: Check inputs
            if batch_idx == 0:
                print(f"Any NaN in x: {torch.isnan(x).any()}, in y: {torch.isnan(y).any()}")

            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = model(x)

                # DEBUG: Check predictions
                if batch_idx == 0 or e == 1:
                    print(f"Any NaN in preds: {torch.isnan(preds).any()}")

                loss = F.mse_loss(preds, y)
                
                # DEBUG: Check loss
                if torch.isnan(loss):
                    print(f"NaN loss at epoch {e}, batch {batch_idx}")
                    print(f"Preds sample: {preds[0]}")
                    print(f"Target sample: {y[0]}")
                    print(f"Loss components: {((preds - y)**2).mean()}")
                    return model

            scaler.scale(loss).backward()
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
                    loss = F.mse_loss(preds, y)
                
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        print(f'Epoch {e:03d} | train: {train_loss: .6f} | val: {val_loss: .6f}')
        progress_bar.set_postfix({'train_loss': round(train_loss, 6), 'val_loss': round(val_loss, 6)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            torch.save(model.state_dict(), 'model/output_mamba/model.pt')
        # lr schduler
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stop at epoch {e}')
                break
        loss_history.append([train_loss, val_loss])
    
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, loss_history

def evaluate(model, data_loader):
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
        'mse': mse,
        'mae': mae,
        'sign_acc': sign_acc
    }

    return metrics, preds, targets

def predict(model, x: torch.Tensor):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        x = x.unsqueeze(0).to(device)
        preds = model(x).cpu().numpy()[0]

    return preds

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

def prepare_data(seq_len=128):
    tech = compute_technical_metrics(window=14)[FEATS['tech']]
    onchain = compute_onchain_features()[FEATS['onchain']]
    poly, _, _ = compute_polymarket_features()
    poly = poly[FEATS['poly']]

    X = pd.concat([tech, onchain, poly], 
                  axis=1, 
                  sort=True).loc['2010-07-18': '2025-12-31'].sort_index().fillna(0)

    Y = compute_btc_returns()[['return_030d', 
                               'return_060d', 
                               'return_090d', 
                               'return_120d']].loc['2010-07-18': '2025-12-31'].sort_index().values

    return X.values, Y

def init_weights(model):
    """Safe initialization that won't crash"""
    
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

def plot_eval(preds, y, col=['30d','60d','90d','120d']):
    fig, ax = plt.subplots(2,2)
    coordinates = [(i, j) for j in range(3) for i in range(2)]
    k = 0
    for l in range(len(col)):
        i, j = coordinates[k]
        sd = np.array(preds[:,l]-y[:,l]).std()
        ax[i][j].plot(preds[:,l], label='preds')
        ax[i][j].fill_between(preds[:,l]-sd, preds[:,l]+sd, alpha=0.2)
        ax[i][j].plot(y[:,l], label='true')
        ax[i][j].set_title(f'Preds Vs True - {col[l]}', fontsize=20)
        ax[i][j].legend(fontsize=16)
        k += 1

    plt.show()

def plot_loss(loss_history):
    plt.figure(figsize=(12,8))
    loss_history = np.array(loss_history)
    plt.plot(loss_history[:, 0], linestyle='--', label='train loss')
    plt.plot(loss_history[:, 1], linestyle='--', label='validation loss')
    plt.title('Train/Validation loss history', fontsize=20)
    plt.legend(fontsize=16)
    plt.show()

def main():
    print('loading data...')
    X, Y = prepare_data()

    train_loader, val_loader, test_loader = load_data(X, Y, batch_size=128)
    seq_len = 128
    num_features = X.shape[1]

    model = CMambaFiveHorizons(
        input_dim=num_features,
        d_model=256,      
        n_layers=4,
        d_state=64,
        d_conv=4,
        expand=2,
        dropout=0.1,
        n_horizons=4,
    )
    model.apply(init_weights)

    print('training...')
    model,loss = train(model, train_loader, val_loader, n_epochs=120, lr=1e-3)
    # model.load_state_dict(torch.load('model/checkpoint/model_2025-12-31.pt'))
    print('evaluating...')
    metrics, preds, targets = evaluate(model, test_loader)
    print("MSE per horizon [30,60,90,120]:", metrics["mse"])
    print("MAE_pct per horizon [30,60,90,120]:", metrics["mae"])
    print("Directional accuracy per horizon[30,60,90,120]:", metrics["sign_acc"])

    n_bootstrap = 100
    bootstran_metrics = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(preds), len(preds), replace=True)
        mse = ((preds[idx] - targets[idx]) ** 2).mean(axis=0)
        bootstran_metrics.append(mse)
    bootstran_metrics = np.array(bootstran_metrics)
    ci_lower = np.percentile(bootstran_metrics, 2.5, axis=0)
    ci_upper = np.percentile(bootstran_metrics, 97.5, axis=0)
    print('MSE per horizon with 95% CI:')
    for i, h in enumerate(['30d', '60d', '90d', '120d']):
        print(f'{h}: {metrics["mse"][i]:.6f} [{ci_lower[i]:.6f}, {ci_upper[i]: .6f}]')

    plot_eval(preds, targets)
    plot_loss(loss_history=loss)

if __name__ == '__main__':
    main()