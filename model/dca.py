from model.feature_selection import prepare_features
from model.utils import compute_btc_returns
import pandas as pd
import numpy as np
import logging
from template.model_development_template import allocate_sequential_stable
from model.feature_selection import prepare_features, compute_btc_returns
from sklearn.linear_model import LinearRegression as lg
from tqdm import tqdm

import logging
from pathlib import Path


FEATS = {
    'tech': ['RSI', 
             'Mayer_multiple',
             'Volume'],
    'poly': ['btc_sentiment'],
    'onchain': ['price_vs_ma',
                'price_ma7_ma30',
                'price_ma7_ma30_gradient',
                'price_ma7_ma30_acceleration',
                'mvrv_zscore',
                'mvrv_zone',
                'mvrv_gradient',
                'AdrBalCnt_ma7_ma30',
                'HashRate_ma7_ma30']
}
START_DATE = '2010-07-18'
logging.basicConfig(level=logging.INFO)

def _prepare_dataset():
    """
    prepare features and targets dataset, align the index
    """
    poly, onchain, tech = prepare_features()
    poly, onchain, tech = poly[FEATS['poly']], onchain[FEATS['onchain']], tech[FEATS['tech']]
    ret = compute_btc_returns()

    X = pd.concat([poly, onchain, tech], axis=1).fillna(0).loc['2010-07-18':'2026-01-13']
    y = ret[['return_030d', 'return_060d', 'return_090d', 'return_182d']].loc[X.index.min():X.index.max()]

    return X, y

def _fit(X: np.ndarray,
        y: np.ndarray) -> tuple:
    """single step fitting"""
    func = lg()
    reg = func.fit(X, y)
    score = func.score(X, y)
    return (reg, score)

def _step(
        X: pd.DataFrame,
        y: pd.DataFrame,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        n_past: int | None = None
) -> pd.DataFrame:
    """
    step prediction on future return
    Args:
        features: selected FEATS from 2010-07-18 to 2025-12-31
        targets: future returns, in 30d, 60d, 90d, 182d
        start: backtest start date
        end: backtest end date
        n_past: number of past days in backtest period
    Returns:
        key: offset (30d, 60d, 90d, 182d)
        value: R2 score, prediction, actual
    """
    if n_past == None:
        n_past = 0
    date_range = pd.date_range(start, end)
    curr_date = date_range[n_past]

    # training dataset is confined in [2010-07-18:current date]
    result = []
    for offset in ['030', '060', '090', '182']:
        # offset the traing dataset to avoid info leak
        pointer = int(offset)
        x_train, y_train = X.loc[:curr_date].iloc[:-pointer], y.loc[:curr_date].iloc[:-pointer][f'return_{offset}d']
        lg_model, R2 = _fit(x_train.values, y_train.values)

        # use leading fitted model (trained by data up to -pointer) to predict future return of today's buy-in 
        y_pred = lg_model.predict(X.loc[curr_date].values.reshape(1,-1))[0]

        result.append({'return_period': f'return_{offset}d',
                        'R2': R2, 
                        'prediction': y_pred, 
                        'actual': y.loc[curr_date][f'return_{offset}d'],
                        'time': curr_date})

    return pd.DataFrame(result).set_index('time')

def _predict_return(
        X: pd.DataFrame, 
        y: pd.DataFrame, 
        end: str | pd.Timestamp = '2025-12-31',
        start: str | pd.Timestamp = '2018-01-01', 
        ) -> tuple:  
    """
    Make future return prediction, and test accuracy/alignment of prediction.
    Return:
        prediction: using _step function and feature data up to current date,
                    predict future (30,60,90,182) returns
        alignment: whether predicted future returns are aligned in sign.
    """  
    n = len(pd.date_range(start, end))
    n_past= 0

    prediction = []
    for t in tqdm(range(n), desc='Predicting'):
        res = _step(X, y, start, end, n_past)
        res['sign_acc'] = np.sign(res['prediction'].to_numpy()) == np.sign(res['actual'].to_numpy())
        res['amount_acc'] = (res['prediction'] - res['actual']) / res['actual']

        prediction.append(res)

        n_past += 1

    prediction = pd.concat(prediction)

    alignment = np.sign(prediction['prediction']).groupby('time').agg('sum').rename('sign_alignment')
    avg_align = pd.DataFrame({
        '2Ups2Downs': (alignment == 0).sum() / len(alignment),
        '3Ups1Down': (alignment == 2).sum() / len(alignment),
        '4Ups': (alignment == 4).sum() / len(alignment),
        '4Downs': (alignment == -4).sum() / len(alignment),
        '1Up3Downs': (alignment == -2).sum() / len(alignment)
    }, index=[(start, end)])

    avg_align.to_csv('data/dca/prediction_alignment.csv')
    prediction = prediction.reset_index().set_index(['return_period', 'time'])

    avg_accuracy = []
    for p in ['030', '060', '090', '182']:
        period = f'return_{p}d'
        avg_acc = prediction.loc[period, 'sign_acc'].cumsum() / (np.arange(n) + 1)
        avg_acc = pd.DataFrame(avg_acc)
        avg_acc['return_period'] = period
        avg_acc = avg_acc.reset_index().set_index(['return_period', 'time'])
        avg_accuracy.append(avg_acc)

    avg_accuracy = pd.concat(avg_accuracy)
    avg_accuracy.to_csv('data/dca/accuracy.csv')
    
    return prediction.sort_index()

# select features to measure strength of reversion force from respective quantile layer
WIN_RATE_FEATS = [
    'HashRate_ma7_ma30',
    'RSI',
    'mvrv_zscore',
    'price_ma7_ma30',    
]

def compute_quantile_winrate(features: pd.DataFrame, 
                             targets: pd.DataFrame,
                             end: str | pd.Timestamp = '2025-12-31',
                             start: str | pd.Timestamp = '2018-01-01') -> tuple:
    """
    Compute win rate (historical return) of features quantiles.
    The result will be used to compute weight multiples.
    The computing uses future return, thus should be masked by leading days when compute multiplier.
    Args:
        X: feature dataframe
        y: btc returns dataframe
    Returns:
        win_rate: btc price return when feature data fall in specified quantile intervals
            index: ['feature', 'return_period', 'time']
            columns: ['<0.1', '0.1-0.2' ...]
            each cell represents last available historical return, by each period (30,60,90,182)
        quantiles: quantile value of each feature
            index: ['quantile' (0.1,0.2 ... 0.9), 'time']
            columns: ['price_vs_ma','price_ma7_ma30','RSI','Mayer_multiple','mvrv_zscore','AdrBalCnt_ma7_ma30','HashRate_ma7_ma30']
            each cell represents the feature's quantile value
    """
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

def compute_quantile_layered_return(feature: pd.DataFrame,
                                    lag_res: pd.DataFrame, 
                                    quantiles: pd.DataFrame, 
                                    start: str | pd.Timestamp = '2018-01-01', 
                                    end: str | pd.Timestamp = '2025-12-31') -> pd.DataFrame:
    """
    compute historical (lagged) return of feature quantile layer
    Args:
        feature: calculated from _prepare_dataset
        lag_res: calculated from compute_quantile_winrate, lagged by (30,60,90,182) days
        quantiles: calclated from compute_quantile_winrate, 9 quantile boudaries per feature per day
    Return:
        quantile_layered_return: 
            each feature value 
                -> (lookup) which quantile layer it belongs to 
                    -> lagged return of this quantile layer
    """

    # lagres: (4 features * 4 return_periods * 4018 days) * 10 quantile intervals
    # quantiles: (9 quantile boundaries * 4018 days) * 4 features
    # lagres, quantiles = compute_quantile_winrate(feature, target)

    # now X is 4018 days * 4 features
    X = feature.loc[pd.date_range(start, end), WIN_RATE_FEATS].copy()
    quantiles = quantiles[WIN_RATE_FEATS].copy()
    
    # quantile lookup tables, each shape 4018 days * 4 features
    lookup_table = [
        X < quantiles.loc[0.1],
        ((X >= quantiles.loc[0.1]) * (X < quantiles.loc[0.2])),
        ((X >= quantiles.loc[0.2]) * (X < quantiles.loc[0.3])),
        ((X >= quantiles.loc[0.3]) * (X < quantiles.loc[0.4])),
        ((X >= quantiles.loc[0.4]) * (X < quantiles.loc[0.5])),
        ((X >= quantiles.loc[0.5]) * (X < quantiles.loc[0.6])),
        ((X >= quantiles.loc[0.6]) * (X < quantiles.loc[0.7])),
        ((X >= quantiles.loc[0.7]) * (X < quantiles.loc[0.8])),
        ((X >= quantiles.loc[0.8]) * (X < quantiles.loc[0.9])),
        X >= quantiles.loc[0.9],
    ]

    quantile_layered_return = []
    # change index order
    lag_res = lag_res.reset_index().set_index(['return_period', 'feature', 'time'])
    for p in ['030', '060', '090', '182']:
        period = f'return_{p}d'
        data = lag_res.loc[period] # 4 * 4018
            
        w = np.zeros_like(X) # 4018 * 4
        for i in range(len(data.columns)):
            interval = data.columns[i]

            # 4 * 4018 -> 4018 * 4
            w_table = data[interval].reset_index().pivot(
                columns = 'feature', # 4 features
                values = interval, # 10 intervals
                index = 'time'
            )
            
            w += w_table * lookup_table[i] 

        w = pd.DataFrame(w, index=pd.date_range(start, end), columns=X.columns)
        w['return_period'] = period
        w = w.reset_index().rename(columns={'index':'time'}).set_index(['return_period','time'])
        quantile_layered_return.append(w)

    quantile_layered_return = pd.concat(quantile_layered_return).sort_index()

    return quantile_layered_return   

def compute_signal(df: pd.DataFrame,
                    start: str | pd.Timestamp = '2018-01-01',
                    end: str | pd.Timestamp = '2025-12-31'):

    # raw signal = predicted return: [(30,60,90,182), 4018 days]
                #   * quantile layered return [(30,60,90,182), 4018 days, 7 features] sum cross features
                #   * prediction alginment of that day [4018, 1].abs()
    # the effect is predicted return (+/-) amplified by exponential quantile force

    # start, end = max(pd.Timestamp(start), pd.Timestamp('2018-01-01')), min(pd.Timestamp(end), pd.Timestamp('2025-12-31'))
    if df.index.min() < pd.Timestamp('2018-01-01'):
        start = pd.Timestamp('2018-01-01')
    if df.index.max() > pd.Timestamp('2025-12-31'):
        end = pd.Timestamp('2025-12-31')
    df = df.loc[start:end]
    df = df.reset_index().set_index(['time', 'return_period'])
    quantile_layered_return = df[WIN_RATE_FEATS]
    prediction = df.drop(WIN_RATE_FEATS, axis=1)
    raw = prediction['prediction']  * (quantile_layered_return.sum(axis=1)) 

    date_range = pd.date_range(start, end)
    signal = np.zeros(len(date_range))
    for p in ['030', '060', '090', '182']:
        period = f'return_{p}d'
        indice = [(d,period) for d in date_range]
        period_raw = raw.loc[indice].clip(lower=0).values.squeeze()
        # period_signal = allocate_sequential_stable(period_raw, len(date_range))
        signal += period_raw
    
    return pd.DataFrame(signal, index=date_range, columns=['signal'])

def compute_weight(_SIGNAL, start, end):
    signal = _SIGNAL.loc[pd.date_range(start, end)].values.squeeze()
    signal = allocate_sequential_stable(signal, len(signal))
    return pd.Series(signal / signal.sum(), index=pd.date_range(start, end), name='weight')
    # return pd.DataFrame(signal / signal.sum(), index=pd.date_range(start, end), columns=['weight'])

# def backtest(start, end):
#     print('loading data...')
#     features, targets = _prepare_dataset()
#     print("making prediction...")

#     prediction = _predict_return(features, targets, end, start)
#     print('separating quantile metrics...')
#     lag_res, quantiles = compute_quantile_winrate(features, targets, end, start)
#     quantile_layered_return = compute_quantile_layered_return(features, lag_res, quantiles, start, end)
#     df = pd.concat([prediction, quantile_layered_return], axis=1).reset_index().set_index('time')
#     _SIGNAL = compute_signal(df, start, end)

#     date_range = pd.date_range(start,end)
#     from template.prelude_template import load_data
#     price = load_data()['PriceUSD_coinmetrics']

#     model_collection, uniform_collection = [], []
#     for i in tqdm(range(len(date_range)-365), desc=f'{start}-{end}-backtesting...'):
#         test_range = date_range[i:i+365]
#         weight = compute_weight(_SIGNAL, test_range[0], test_range[-1])               
#         model_collection.append((1e6 * weight / price.loc[test_range]).sum())
#         uniform_collection.append((1e6 * (np.ones(len(test_range)) / len(test_range)) / price.loc[test_range]).sum())
    
#     result = pd.DataFrame({'model_collection': model_collection, 'uniform_collection':uniform_collection})

#     print(f'win: {(result['model_collection'] > result['uniform_collection']).sum()}')

#     return result

