from model.mamba import prepare_data, predict, CMamba
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from template.model_development_template import allocate_sequential_stable
from model.final_model_prelude_template import load_data
from model.final_model_backtest_template import run_full_analysis
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

WIN_RATE_FEATS = [
    'HashRate_ma7_ma30',
    'mvrv_zscore',
    'RSI', 
    'Mayer_multiple', 
]

def compute_quantile_winrate(features: pd.DataFrame, 
                             targets: pd.DataFrame,
                             end: str | pd.Timestamp = '2025-12-31',
                             start: str | pd.Timestamp = '2018-01-01') -> tuple:
    date_range = pd.date_range(start, end)
    n_past = 0

    all_res, quantiles = [], []
    for t in tqdm(range(len(date_range)), desc='Quantile separating'):
        curr_date = date_range[n_past]

        # only use data up to current date
        X = features.loc[:curr_date][WIN_RATE_FEATS]
        y = targets.loc[:curr_date]
        interval = int(np.ceil(len(X)/10))

        res = []
        for col in WIN_RATE_FEATS:
            # get feature index of each quantile interval
            s = X[col].sort_values()
            indice = {
                '<0.1': s.iloc[:interval].index,
                '0.1-0.2': s.iloc[interval:2*interval].index,
                '0.2-0.3': s.iloc[2*interval:3*interval].index,
                '0.3-0.4': s.iloc[3*interval:4*interval].index,
                '0.4-0.5': s.iloc[4*interval:5*interval].index,
                '0.5-0.6': s.iloc[5*interval:6*interval].index,
                '0.6-0.7': s.iloc[6*interval:7*interval].index,
                '0.7-0.8': s.iloc[7*interval:8*interval].index,
                '0.8-0.9': s.iloc[8*interval:9*interval].index,
                '>=0.9': s.iloc[9*interval:].index
            }

            # for each interval, get win_rate (future return)
            win_rate = pd.DataFrame({k: y.loc[indice[k]].mean() for k in indice.keys()}).reset_index()
            win_rate.rename(columns={'index':'return_period'}, inplace=True)
            win_rate['feature'] = col
            res.append(win_rate)

        res = pd.concat(res).reset_index().drop('index', axis=1)
        res['time'] = curr_date 
        all_res.append(res)

        # get quantile boundary using feature data up to current day
        q = X.quantile(np.arange(1,10)/10).reset_index().rename(columns={'index': 'quantile'})
        q['time'] = curr_date
        quantiles.append(q)

        n_past += 1

    all_res = pd.concat(all_res).set_index(['feature', 'return_period', 'time'])
    quantiles = pd.concat(quantiles).set_index(['quantile', 'time'])

    # We should lag 'all_res' to prevent info leak since it was obtained by future return
    # The quantile data need not to be lagged since they are daily self-descriptive data 
    lag_res = []
    for offset in ('030', '060', '090', '182'):
        pointer, return_period = int(offset), f'return_{offset}d'
        for feature in WIN_RATE_FEATS:
            subset = all_res.loc[feature].loc[return_period].shift(pointer).bfill().reset_index()
            subset['feature'] = feature
            subset['return_period'] = return_period
            subset = subset.set_index(['feature', 'return_period', 'time'])
            lag_res.append(subset)
    
    lag_res = pd.concat(lag_res)
    lag_res.to_csv('data/dca/feature_lag_return.csv')
    quantiles.to_csv('data/dca/feature_quantiles.csv')
    return lag_res, quantiles 

def post_process_signal(_Signal,
                        start='2018-01-01', 
                        end='2025-12-31',
                        qa=0.1,
                        qb=0.9):
    
    from model.LinReg import _prepare_dataset
    from model.utils import compute_btc_returns
    x, y = _prepare_dataset()
    _, quantiles = compute_quantile_winrate(x, y)
    res = compute_btc_returns()
    res_std_030d = res['return_030d'].rolling(3).std().bfill().shift(1).loc['2018-01-01':]

    features = x.loc[start: end].copy() # type: ignore    
    q_hashrate = quantiles.loc[[qa,qb], 'HashRate_ma7_ma30']
    q_rsi = quantiles.loc[[qa,qb], 'RSI']

    mamba_signals = _Signal.copy()    
    base_weight = np.ones(365) / 365
    window_start = pd.date_range('2018-01-01', '2024-12-31')
    post_processed_signals = {}
    
    pbar = tqdm(range(len(window_start)), desc='Post Processing Signals...')
    for idx, d in enumerate(pbar):
        day = window_start[idx]
        window_end = day + pd.offsets.DateOffset(364)

        signals = base_weight.copy()

        if mamba_signals is not None:
            mamba_signal_window = mamba_signals[day.strftime('%Y-%m-%d')]
            signals = ((np.abs(mamba_signal_window) >= 0.8 * res_std_030d.loc[day]) * mamba_signal_window * 200 + 1) * base_weight
            signals = np.clip(signals, a_min=0, a_max=15)

        if x.loc[day, 'Mayer_multiple'] <= 1.2:  #res_std_030d.loc[day] >= 0.025 and       
            q1_hashrate = -.01 * (features.loc[day:window_end, 'HashRate_ma7_ma30'].values <= q_hashrate.loc[qa, day:window_end].values)
            q9_hashrate = 30 * (features.loc[day:window_end, 'HashRate_ma7_ma30'].values >= q_hashrate.loc[qb, day:window_end].values)

            # q1_rsi = -.01 * (features.loc[day:window_end, 'RSI'].values <= q_rsi.loc[qa, day:window_end].values)
            q9_rsi = 30 * (features.loc[day:window_end, 'RSI'].values >= q_rsi.loc[qb, day:window_end].values)

            signals = signals + q9_rsi + q9_hashrate + q1_hashrate # + q1_rsi
            signals = np.clip(signals, a_min=0, a_max=15) 

        signals = allocate_sequential_stable(signals, len(signals))
        post_processed_signals[day.strftime('%Y-%m-%d')] = signals
    return post_processed_signals

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

    if os.path.exists('data/dca/final_model_signals.json'):
        with open ('data/dca/final_model_signals.json', 'r') as f:
            signal = json.load(f)
    else:
        print('Signal doesnt exist! run compute_signal first!')
        signal = compute_signal()
        signal = post_process_signal(signal)

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
    output_dir = base_dir / "output_final"

    if os.path.exists('data/dca/final_model_signals.json'):
        with open ('data/dca/final_model_signals.json', 'r') as f:
            _SIGNAL = json.load(f)
            _SIGNAL = post_process_signal(_SIGNAL)
    else:
        _SIGNAL = compute_signal()
        _SIGNAL = post_process_signal(_SIGNAL)
        _SIGNAL = {k:v.tolist() for k,v in _SIGNAL.items()}
        with open('data/dca/final_model_signals.json', 'w') as file:
            json.dump(_SIGNAL, file) # type: ignore

    run_full_analysis(
        btc_df=btc_df,
        signal_dict = _SIGNAL,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="DCA (Final)"
    )

if __name__ == '__main__':
    main()


def fast_backtest(start='2018-01-01',
                  end='2025-12-31',
                  qa=0.1,
                  qb=0.9):
    
    from model.LinReg import _prepare_dataset
    from model.utils import compute_btc_returns
    x, y = _prepare_dataset()
    _, quantiles = compute_quantile_winrate(x, y)
    res = compute_btc_returns()
    res_std_030d = res['return_030d'].rolling(3).std().bfill().shift(1).loc['2018-01-01':]
    price = res['price']

    features = x.loc[start: end].copy() # type: ignore    
    q_hashrate = quantiles.loc[[qa,qb], 'HashRate_ma7_ma30']
    q_rsi = quantiles.loc[[qa,qb], 'RSI']

    with open ('data/dca/mamba_signals.json', 'r') as f:  
        mamba_signals = json.load(f)

    base_weight = np.ones(365) / 365
    window_start = pd.date_range('2018-01-01', '2024-12-31')
    
    win, history = 0, []
    pbar = tqdm(range(len(window_start)), desc='Post Processing Signals...')
    for idx, d in enumerate(pbar):
        day = window_start[idx]
        window_end = day + pd.offsets.DateOffset(364)

        signals = base_weight.copy()

        mamba_signal_window = mamba_signals[day.strftime('%Y-%m-%d')]
        signals = ((np.abs(mamba_signal_window) >= 0.8 * res_std_030d.loc[day]) * mamba_signal_window * 200 + 1) * base_weight
        signals = np.clip(signals, a_min=0, a_max=15)

        if x.loc[day, 'Mayer_multiple'] <= 1.2:   #res_std_030d.loc[day] >= 0.0005 and      
            q1_hashrate = -.1 * (features.loc[day:window_end, 'HashRate_ma7_ma30'].values <= q_hashrate.loc[qa, day:window_end].values)
            q9_hashrate = 30* (features.loc[day:window_end, 'HashRate_ma7_ma30'].values >= q_hashrate.loc[qb, day:window_end].values)

            # q1_rsi = -.01 * (features.loc[day:window_end, 'RSI'].values <= q_rsi.loc[qa, day:window_end].values)
            q9_rsi = 30 * (features.loc[day:window_end, 'RSI'].values >= q_rsi.loc[qb, day:window_end].values)

            signals = signals + q9_rsi + q9_hashrate + q1_hashrate # + q1_rsi
            signals = np.clip(signals, a_min=0, a_max=15) 

        signals = allocate_sequential_stable(signals, len(signals))

        model_collection = (1e6 * signals / price.loc[day:window_end]).sum()
        uniform_collection = (1e6 * base_weight / price.loc[day:window_end]).sum()

        if model_collection > uniform_collection:
            win += 1
        history.append([model_collection, uniform_collection])

        pbar.set_description(f'{win}/{idx+1}, model:{model_collection}, uniform:{uniform_collection}')

    history = np.array(history)
    print(f'surplus:{history[:,0].sum()- history[:,1].sum()}')
    plt.fill_between(window_start, 0, (history[:,0]-history[:,1]).cumsum())
    plt.title('cumulated surplus')
    plt.show()
    



