from model.mamba import prepare_data, predict, CMamba
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from template.model_development_template import allocate_sequential_stable
from model.prelude_template_mamba import load_data
from model.backtest_template_mamba import run_full_analysis
from tqdm import tqdm
import logging
from pathlib import Path
import os, json
import matplotlib.pyplot as plt
import seaborn as sns


START = '2010-07-18'
model_list = ['2018-12-31', '2019-06-30', '2019-12-31', '2020-06-30', '2020-12-31', 
            '2021-06-30', '2021-12-31', '2022-06-30', '2022-12-31', '2023-06-30', '2023-12-31',
            '2024-06-30', '2024-12-31', '2025-06-30', '2025-12-31']
SEQ_LEN=128
X, _ = prepare_data()

# refit the deterministic RobustScaler
def feature_scaler(X: np.ndarray,
                    val_ratio: float = 0.2,
                    test_ratio: float = 0.1):
    """
    Since RobustScaler is deterministic, we can retrieve scalers use same x_train.
    """    
    N = X.shape[0]
    n_test = max(int(N * test_ratio), 365)
    n_val = int(N * val_ratio)
    n_train = N - n_val - n_test

    X_train = X[:n_train]
    X_scaler = RobustScaler(quantile_range=(2.5, 97.5))
    # scalers only be fitted on x_train
    X_scaler.fit(X_train)

    return X_scaler

def retrieve_scalers():
    """
    For total of 15 trained models (last date of each in variable 'model_list'),
    retrieve scaler of each model
    """
    scalers = []
    dataset_range = pd.date_range('2010-07-18', '2025-12-31')
    for end_date in model_list:
        end_idx = dataset_range.tolist().index(pd.Timestamp(end_date))

        # X is computed from function prepare_data, and the type is np.ndarray
        # Since X is np.ndarray, the slice of it should up to end_indx + 1 
        # All features should start from '2010-07--18'
        features = X[:end_idx + 1]
        scalers.append(feature_scaler(features))
    return scalers

def make_batch(X, seq_len=SEQ_LEN):
    """
    Every target is predicted by previous 128 steps features.
    """
    T = len(X)
    XX = []
    for t in range(seq_len, T):
            XX.append(X[t-seq_len: t])
    XX = torch.tensor(np.stack(XX), dtype=torch.float32)   
    return XX

def compute_signal() -> dict:
    """
    Compute all sinals from 2018-01-01 to 2025-12-31
    1. Models are trainned in a way preventing leakage:
        The last 10% time steps (365 at least) is test data. Test data never involved in training and validation.
        The validation data is 20% time steps before test data, and mask 120 last time steps.
        The remaining is training data, which is no more than 70% of total time steps. 
    2. Test windows map models:
        end_date < 2018-12-31 --> model_2018-12-31.pt
        end_date < 2019-06-30 --> model_2019-06-30.pt
        end_date < 2019-12-31 --> model_2019-12-31.pt
        ...
    3. Returned signals are in format: {start_date:np.ndarray of 365 signals}:
        {'2018-01-01': [365 signals]}
        {'2018-01-02': [365 signals]}
        ...
    4. Since the signals are computed in a rolling manner, below backtest functions are revised:
        backtest_template_mamba.py --> copied from backtest_template.py
            (run_full_analysis)
        prelude_template_mamba.py --> copied from prelude_template.py
            (check_strategy_submission_ready, backtest_dynamic_dca, compute_cycle_spd)
    """
    start_date = '2018-01-01'
    end_date = '2025-12-31'    
    # 1. load features in date range
    # the mamba model use leading 128 timestep data to predict current date signal
    # and that's the reason I should retrieve feature dataset instead of using df_window
    # all feature data start from 2010-07-18 to match RobustScalers used during training
    print('loding data...')
    X, _ = prepare_data('2010-07-18', end_date)

    # 2. load model correspondent to the date range
    # the models were trained on a preventing data leakage manner
    print('loading model...model/checkpoint/model_2018-12-31.pt')
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
    model.load_state_dict(torch.load('model/checkpoint/model_2018-12-31.pt')) # the first model
    
    # 3. retrieve RobustScalers used during training
    dataset_range = pd.date_range('2010-07-18', '2025-12-31')
    scalers = retrieve_scalers()
    assert len(scalers) == len(model_list)

    # 4. compute signal in a rolling manner, since I trained models per six months
    results, curr_model = {}, 0
    test_range = pd.date_range(start_date, end_date)
    pbar = tqdm(range(len(test_range)-365), desc='Predicting signals...')    
    for d_idx, d in enumerate(pbar):
        curr_range = test_range[d: d+365]
        # load model corresponded to current range
        model_idx = (curr_range[-1].strftime('%Y-%m-%d') <= np.array(model_list)).tolist().index(True) 
        if model_idx > curr_model:
            model_name = f'model/checkpoint/model_{model_list[model_idx]}.pt'
            print(f'loading model...{model_name}')            
            model.load_state_dict(torch.load(model_name))
            curr_model += 1
        
        # current date signal is predicted by previous 128 time step data
        X_test = X[dataset_range.tolist().index(curr_range[0])-128: dataset_range.tolist().index(curr_range[-1])+1]
        X_test_scaled = torch.tensor(scalers[model_idx].transform(X_test), dtype=torch.float32)
        X_test_scaled = make_batch(X_test_scaled)

        uniform_weight = np.ones(365) / 365

        # Use predicted signal as booster, dynamically adjust uniform weight
        # The only Hyperparameter is the booster threshold:
        #   When absolute value of predicted signal > 0.05, we believe it is a real signal
        signal = []
        for x in X_test_scaled:
            signal.append(predict(model, x)[-1])
        signal = np.array(signal)
        signal = ((np.abs(signal) >= 0.05) * signal * 2 + 1) * uniform_weight
        signal = np.clip(signal, a_min=0, a_max=5)        
        signal = allocate_sequential_stable(signal, len(signal))
        assert (signal.sum() - 1) < 1e-4

        # the signal is a dictionary with keys are the first day of current range
        results[curr_range[0].strftime('%Y-%m-%d')] = signal.tolist()    

    return results

def compute_weights_wrapper(df_window: pd.DataFrame) -> np.ndarray:
    """Wrapper for compute weights.    
    Important revision:
        Output is ndarray instead of dataframe.
    """
    if df_window.empty:
        start_date, end_date = pd.Timestamp('2018-01-01'), pd.Timestamp('2025-12-31')
    else:
        start_date = df_window.index.min()
        end_date = df_window.index.max()

    if os.path.exists('data/dca/mamba_signals.json'):
        with open ('data/dca/mamba_signals.json', 'r') as f:
            signal = json.load(f)
    else:
        print('Signal doesnt exist! run compute_signal first!')
        signal = compute_signal()
    
    weight = np.array(signal[start_date.strftime('%Y-%m-%d')])

    return weight / weight.sum()

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info("Starting Bitcoin DCA Strategy Analysis - DCA (mamba)")

    btc_df = load_data().loc[pd.date_range('2018-01-01', '2025-12-31')]
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output_mamba"

    if os.path.exists('data/dca/mamba_signals.json'):
        with open ('data/dca/mamba_signals.json', 'r') as f:
            _SIGNAL = json.load(f)
    else:
        _SIGNAL = compute_signal()
        with open('data/dca/mamba_signals.json', 'w') as file:
            json.dump(_SIGNAL, file) # type: ignore

    run_full_analysis(
        btc_df=btc_df,
        signal_dict = _SIGNAL,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="DCA (Mamba)"
    )

if __name__ == '__main__':
    main()


def custom_backtest():
    """
    customized backtest just to check the win rate against uniform strategy
    """

    print('loading data...')
    test_start, test_end = '2018-01-01', '2025-12-31'
    test_range = pd.date_range(test_start, test_end)

    # load features from '2010-07-18' to '2025-12-31'
    X, _ = prepare_data(START, test_end)
    dataset_range = pd.date_range(START, test_end)
    price = load_data().loc[START:test_end, 'PriceUSD_coinmetrics']
    scalers = retrieve_scalers()
    assert len(scalers) == len(model_list)

    print('loading model...model/checkpoint/model_2018-12-31.pt')
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
    model.load_state_dict(torch.load('model/checkpoint/model_2018-12-31.pt'))

    results, curr_model, win, window_start = [], 0, 0, []
    pbar = tqdm(range(len(test_range)-365), desc='Testing...')
    for d_idx, d in enumerate(pbar):
        curr_range = test_range[d: d+365]

        # select model based on end date of current range
        model_idx = (curr_range[-1].strftime('%Y-%m-%d') <= np.array(model_list)).tolist().index(True) 
        if model_idx > curr_model:
            model_name = f'model/checkpoint/model_{model_list[model_idx]}.pt'
            print(f'loading model...{model_name}')            
            model.load_state_dict(torch.load(model_name))
            curr_model += 1
        
        # prepare data structure: (1) 128 sequence length, (2) scaling
        X_test = X[dataset_range.tolist().index(curr_range[0])-128: dataset_range.tolist().index(curr_range[-1])+1]
        X_test_scaled = scalers[model_idx].transform(X_test)
        X_test_scaled = make_batch(X_test_scaled)

        # Assume capital budget is 1e6, compare cumulated bitcoin collections between 2 strategies
        # Uniform strategy
        price_curr_range = price.loc[curr_range]
        uniform_weight = np.ones(365) / 365
        uniform_collection = ((1e6 * uniform_weight) / price_curr_range).sum()

        # Prediction boosting strategy
        signal = []
        for x in X_test_scaled:
            signal.append(predict(model, x)[-1])
        signal = np.array(signal)
        signal = ((np.abs(signal) >= 0.05) * signal * 2 + 1) * uniform_weight
        signal = np.clip(signal, a_min=0, a_max=5)        
        signal = allocate_sequential_stable(signal, len(signal))
        assert (signal.sum() - 1) < 1e-4
        capital = 1e6 * signal
        model_collection = (capital / price_curr_range).sum()

        if model_collection > uniform_collection:
            win += 1

        results.append([model_collection, uniform_collection])
        window_start.append(curr_range[0])
        pbar.set_description(f'model: {model_collection:.4f} | unif: {uniform_collection:.4f} | win_rate: {win}/{d+1} | end data: {curr_range[-1]}')

    results = np.array(results)
    print(f'num_backtests: {len(results)} | win: {(results[:,0] > results[:,1]).sum()} | surplus: {results[:,0].sum() - results[:,1].sum()}')
    print(results, 'sum', results.sum(axis=0))

    results = pd.DataFrame({'mamba': results[:, 0], 'uniform': results[:, 1]}, index=window_start)
    results = results.reset_index().rename(columns={'index':'window_start'})
    wins = 1 * (results['mamba'] > results['uniform'])
    loses = -1 * (results['mamba'] < results['uniform'])
    results['win/lose/tie'] = 1 * (wins + loses)
    results['type'] = results['win/lose/tie'].map({1:'WIN', 0:'TIE', -1:'LOSE'})

    plt.figure(figsize=(20,8))
    sns.barplot(data=results[['window_start', 'win/lose/tie', 'type']],
                x='window_start',
                y='win/lose/tie',
                hue='type',
                palette={'WIN':'green', 'TIE':'orange', 'LOSE':'red'},
                alpha=0.6)
    for day in model_list:
        plt.axvline((pd.Timestamp(day) + pd.offsets.DateOffset(-364)).strftime('%Y-%m-%d'),
                    ymax=1.3,
                    ymin=-1.3,
                    linestyle='--')
    ties = results[results['type'] == 'TIE']
    plt.scatter(ties['window_start'].map(lambda x: x.strftime('%Y-%m-%d')),
                ties['win/lose/tie'],
                s=0.15,
                c='orange',
                marker='*')
    xticks = ['2018-01-01','2018-06-30'] + model_list
    plt.xticks(ticks=xticks)
    plt.title('Mamba Vs Uniform Competing History')
    plt.show()

    return results
