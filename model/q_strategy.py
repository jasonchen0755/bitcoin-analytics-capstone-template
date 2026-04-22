from model.LinReg import _prepare_dataset, compute_btc_returns
from model.utils import plot_extreme_quantile_return, compute_btc_returns
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json
from template.model_development_template import allocate_sequential_stable
from template.prelude_template import load_data


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

x, y = _prepare_dataset()
lag_res, quantiles = compute_quantile_winrate(x, y)
res = compute_btc_returns()
res_std_030d = res['return_030d'].rolling(3).std().bfill().shift(1).loc['2018-01-01':]
# feat_name = 'AdrBalCnt_ma7_ma30'
 # plot_extreme_quantile_return(x, quantiles, res, '2018-01-01', '2025-12-31', feat_name)

def main(start='2018-01-01', 
         end='2025-12-31',
         qa=0.1,
         qb=0.9):
    features = x.loc[start: end].copy() # type: ignore
    price = load_data().loc[start:end, 'PriceUSD_coinmetrics']
    
    # res = res.loc[start, end].copy() # type: ignore
    q_hashrate = quantiles.loc[[qa,qb], 'HashRate_ma7_ma30']
    q_rsi = quantiles.loc[[qa,qb], 'RSI']

    mamba_signals = None
    if os.path.exists('data/dca/mamba_signals.json'):
        with open ('data/dca/mamba_signals.json', 'r') as f:
            mamba_signals = json.load(f)
    
    base_weight = np.ones(365) / 365

    results, win = [], 0
    window_start = pd.date_range('2018-01-01', '2024-12-31')
    
    pbar = tqdm(range(len(window_start)), desc='Back Testing...')
    for idx, d in enumerate(pbar):
        day = window_start[idx]
        window_end = day + pd.offsets.DateOffset(364)
        curr_range = pd.date_range(day, window_end)
        # day = day.strftime('%Y-%m-%d')

        signals = base_weight.copy()
        price_curr_range = price.loc[curr_range]
        uniform_collection = ((1e6 * base_weight) / price_curr_range).sum()

        if mamba_signals is not None:
            mamba_signal_window = mamba_signals[day.strftime('%Y-%m-%d')]
            signals = ((np.abs(mamba_signal_window) >= 0.05) * mamba_signal_window * 200 + 1) * base_weight
            signals = np.clip(signals, a_min=0, a_max=15)

        if res_std_030d.loc[day] >= 0.025 and x.loc[day, 'Mayer_multiple'] <= 1.3:        
            q1_hashrate = -.01 * (features.loc[day:window_end, 'HashRate_ma7_ma30'].values <= q_hashrate.loc[qa, day:window_end].values)
            q9_hashrate = .05 * (features.loc[day:window_end, 'HashRate_ma7_ma30'].values >= q_hashrate.loc[qb, day:window_end].values)

            q1_rsi = -.01 * (features.loc[day:window_end, 'RSI'].values <= q_rsi.loc[qa, day:window_end].values)
            q9_rsi = .05 * (features.loc[day:window_end, 'RSI'].values >= q_rsi.loc[qb, day:window_end].values)

            signals = signals + q9_rsi + q1_rsi + q9_hashrate + q1_hashrate 

            signals = np.clip(signals, a_min=0, a_max=15) 

        signals = allocate_sequential_stable(signals, len(signals))
        assert (signals.sum() - 1) < 1e-4
        capital = 1e6 * signals
        model_collection = (capital / price_curr_range).sum()

        if model_collection > uniform_collection:
            win += 1

        results.append([model_collection, uniform_collection])

        pbar.set_description(f'model: {model_collection:.4f} | unif: {uniform_collection:.4f} | win_rate: {win}/{d+1} | end data: {curr_range[-1]}')

    results = np.array(results)
    print(f'num_backtests: {len(results)} | win: {(results[:,0] > results[:,1]).sum()} | surplus: {results[:,0].sum() - results[:,1].sum()}')
    print(results, 'sum', results.sum(axis=0))
    fig, ax = plt.subplots(1,2, figsize=(20,6))
    ax[0].plot(pd.date_range('2018-01-01', '2024-12-31'), results[:,0], label='model collection', linestyle='--')
    ax[0].plot(pd.date_range('2018-01-01', '2024-12-31'), results[:,1], label='uniform collection', linestyle='--')
    ax[0].legend()
    ax[1].fill_between(pd.date_range('2018-01-01', '2024-12-31'), 0, (results[:,0] - results[:,1]).cumsum())
    plt.show()

if __name__ == '__main__':
    main()
