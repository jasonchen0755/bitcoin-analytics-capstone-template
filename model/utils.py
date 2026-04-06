import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return ((series - mean) / std).fillna(0)

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    ex = np.exp(x - x.max())
    return ex / ex.sum()

def load_yf():
    """
    load btc exchange trading data from yahoo finance
    data start from 9/17/2014
    """
    file_path = './data/yfinance/BTC-USD.csv'
    df = pd.read_csv(file_path)
    df.rename(columns={'Date':'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time']).dt.normalize()
    df.set_index('time', inplace=True)
    return df

def compute_technical_metrics(window: int=14) -> pd.DataFrame:
    """
    args:
        btc: pandas dataframe with open, high, low, close, volume of BTC trading data
        window: window size for RSI and ATR smoothing
    returns:
        pd.DataFrame, additional columns:
            RSI: the ratio of average gain to average loss
            ATR: average true range / close
            PIC: sma_111 - ema_350 * 2
            Mayer_Multiple: price / sma_200
        we will lag technical metrics by 1 day to prevent info leak
    """
    btc = load_yf()
    close = btc['Close']
    # PIC
    sma_111 = close.rolling(window=111).mean().fillna(0)
    sma_350 = close.rolling(window=350).mean().fillna(0)
    btc['PIC'] = sma_111 - 2* sma_350

    # RSI
    delta = close.diff(1)
    gain = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/window, adjust=False).mean()
    rsi = gain / loss
    btc['RSI'] = 100 - (100 / (1+rsi))

    # ATR
    high_low = btc['High'] - btc['Low']
    high_close = (btc['High'] - btc['Close'].shift(1)).abs()
    low_close = (btc['Low'] - btc['Close'].shift(1)).abs()
    atr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    btc['ATR'] = atr.rolling(window=window).mean() 

    # Mayer_Multiple
    sma_200 = close.rolling(window=200).mean()
    btc['Mayer_multiple'] = close / sma_200       

    return btc.shift(1).fillna(0)

def compute_ma7_ma30(series:pd.Series) -> pd.Series:
    """
    compute reversion force which push ma7 towards ma30
    we will lag features to prevent info leak
    """
    ma7 = series.rolling(window=7, min_periods=3).mean()
    ma30 = series.rolling(window=30, min_periods=15).mean()
    res = pd.Series(((ma7 / ma30) - 1).clip(-1, 1).fillna(0), name=f'{series.name}_ma7_ma30')
    
    return res.shift(1).fillna(0)

def compute_onchain_features():
    """
    1.Wrap onchain features from trilemma example
    2.Add reversion force which push ma7 towards ma30 for below columns:
        HashRate, TxTfrCnt, AdrBalCnt, FeeTotNtv, PriceUSD_coinmetrics
    """
    from example_1.model_development_example_1 import precompute_features
    from template.prelude_template import load_data

    coinmetrics = load_data()
    # function 'precompute_features' provided by trilemma lag features by 1 day
    onchain_features = precompute_features(coinmetrics[['PriceUSD_coinmetrics', 'CapMVRVCur']])

    # function 'compute_ma7_ma30' lags output by 1 day, align with above results
    coinmetrics = coinmetrics[['HashRate', 'TxTfrCnt', 'AdrBalCnt', 'FeeTotNtv', 'PriceUSD_coinmetrics']]
    onchain_features['HashRate_ma7_ma30'] = compute_ma7_ma30(coinmetrics['HashRate']).loc[onchain_features.index]
    onchain_features['TxTfrCnt_ma7_ma30'] = compute_ma7_ma30(coinmetrics['TxTfrCnt']).loc[onchain_features.index]
    onchain_features['AdrBalCnt_ma7_ma30'] = compute_ma7_ma30(coinmetrics['AdrBalCnt']).loc[onchain_features.index]
    onchain_features['FeeTotNtv_ma7_ma30'] = compute_ma7_ma30(coinmetrics['FeeTotNtv']).loc[onchain_features.index]
    onchain_features['price_ma7_ma30'] = compute_ma7_ma30(coinmetrics['PriceUSD_coinmetrics']).loc[onchain_features.index]
    onchain_features['price_ma7_ma30_gradient'] = onchain_features['price_ma7_ma30'].diff(1).fillna(0)
    onchain_features['price_ma7_ma30_acceleration'] = onchain_features['price_ma7_ma30_gradient'].diff(1).fillna(0)
    onchain_features['price_gradient'] = onchain_features['PriceUSD_coinmetrics'].diff(1).fillna(0)
    onchain_features['price_acceleration'] = onchain_features['price_gradient'].diff(0).fillna(0)

    onchain_features.drop(['PriceUSD_coinmetrics', 'price_ma'], axis=1, inplace=True)

    return onchain_features.fillna(0)

def compute_polymarket_features():
    """
    1. Wrap polymarket_sentiment from trilemma example
    2. Add federal rate relevant markets data as features:
        rate_up: polymarket betting fed will raise rate;
        rate_down: polymarket betting fed will cut rate;
    """
    from template.prelude_template import load_polymarket_data, load_data
    polymarkets = load_polymarket_data()['markets']

    # separate rate betting up or down
    rate_markets = polymarkets[polymarkets['question'].str.contains('rate', case=False, na=False)].copy()
    rate_markets['created_at'] = pd.to_datetime(rate_markets['created_at']).dt.normalize()
    rate_up = rate_markets[rate_markets['question'].str.contains('raise|increase|above|up|higher', case=False, na=False)]
    rate_up = rate_up[rate_up['question'].str.contains('fed', case=False, na=False)]
    rate_up = rate_up[~rate_up['question'].str.contains('no change', case=False, na=False)].copy()
    rate_down = rate_markets[rate_markets['question'].str.contains('cut|decrease|below|down|lower', case=False, na=False)]
    rate_down = rate_down[rate_down['question'].str.contains('fed', case=False, na=False)]
    rate_down = rate_down[~rate_down['question'].str.contains('no change', case=False, na=False)].copy()

    def normalize(series:pd.DataFrame):
        """normalize to z score then softmax to get non-negative features"""
        daily_stats = series.groupby('created_at').agg(
            daily_market_count = ('market_id', 'count'),
            daily_volume = ('volume', 'sum')
        ).reset_index()
        vol_mu = daily_stats['daily_volume'].mean()
        vol_std = daily_stats['daily_volume'].std()
        daily_stats['daily_volume_z'] = (daily_stats['daily_volume'] - vol_mu) / vol_std
        daily_stats['daily_volume_z'] = np.clip(daily_stats['daily_volume_z'], a_min=-4, a_max=4)
        daily_stats['daily_volume_z_exp'] = softmax(daily_stats['daily_volume_z'].to_numpy())
        return daily_stats
    
    rate_up = normalize(rate_up)
    rate_up = rate_up.rename(columns={'created_at':'time'}).set_index('time').sort_index()
    rate_down = normalize(rate_down)
    rate_down = rate_down.rename(columns={'created_at':'time'}).set_index('time').sort_index()

    from example_1.model_development_example_1 import precompute_features
    coinmetrics = load_data()
    
    # function 'precompute_features' provided by trilemma lag features by 1 day
    btc_sentiment = precompute_features(coinmetrics[['PriceUSD_coinmetrics', 'CapMVRVCur']])['polymarket_sentiment']
    features = pd.DataFrame({
        # 'btc_sentiment': btc_sentiment,
        'rate_up_market_count': pd.Series(0, index=btc_sentiment.index),
        'rate_down_market_count': pd.Series(0, index=btc_sentiment.index),
        'rate_up_volume_z_exp': pd.Series(0.0, index=btc_sentiment.index),
        'rate_down_volume_z_exp': pd.Series(0.0, index=btc_sentiment.index)
    }, index=btc_sentiment.index)
    features.loc[rate_up.index, 'rate_up_market_count'] = rate_up['daily_market_count']
    features.loc[rate_down.index,'rate_down_market_count'] = rate_down['daily_market_count']
    features.loc[rate_up.index, 'rate_up_volume_z_exp'] = rate_up['daily_volume_z_exp']
    features.loc[rate_down.index, 'rate_down_volume_z_exp'] = rate_down['daily_volume_z_exp']

    # lag
    features = features.shift(1)
    # combinde already shifted btc_sentiment
    features['btc_sentiment'] = btc_sentiment # already shifted
 
    return features.fillna(0), rate_down, rate_up

def compute_btc_returns():
    """
    compute BTC 1, 7, 14, 30, 60, 90, 182, 365 days return
    """
    from template.prelude_template import load_data
    btc = load_data()
    price = btc['PriceUSD_coinmetrics']

    return pd.DataFrame({
        'price': price,
        'return_001d': price.pct_change(periods=1, fill_method=None).fillna(0),
        'return_007d': price.pct_change(periods=7, fill_method=None).fillna(0),
        'return_014d': price.pct_change(periods=14, fill_method=None).fillna(0),
        'return_030d': price.pct_change(periods=30, fill_method=None).fillna(0),
        'return_060d': price.pct_change(periods=60, fill_method=None).fillna(0),
        'return_090d': price.pct_change(periods=90, fill_method=None).fillna(0),
        'return_182d': price.pct_change(periods=182, fill_method=None).fillna(0),
        'return_365d': price.pct_change(periods=365, fill_method=None).fillna(0)
    }, index = price.index).sort_index()

def plot_prediction(prediction: pd.DataFrame):
    pred_30d_ma5 = prediction.loc['return_030d', 'prediction'].rolling(5).mean().bfill()
    pred_60d_ma10 = prediction.loc['return_060d', 'prediction'].rolling(10).mean().bfill()
    pred_90d_ma15 = prediction.loc['return_090d', 'prediction'].rolling(15).mean().bfill()
    pred_182d_ma30 = prediction.loc['return_182d', 'prediction'].rolling(30).mean().bfill()

    true_30d = prediction.loc['return_030d', 'actual']
    true_60d = prediction.loc['return_060d', 'actual']
    true_90d = prediction.loc['return_090d', 'actual']
    true_182d = prediction.loc['return_182d', 'actual']

    fig, ax = plt.subplots(2,2)
    ax[0][0].plot(pred_30d_ma5, label='pred-30d-ma5')
    ax[0][0].plot(true_30d, label='true_30d')
    ax[0][0].set_title('pred Vs true return - 30d')
    ax[0][0].legend()
    
    ax[0][1].plot(pred_60d_ma10, label='pred-60d-ma10')
    ax[0][1].plot(true_60d, label='true_60d')
    ax[0][1].set_title('pred Vs true return - 60d')
    ax[0][1].legend()

    ax[1][0].plot(pred_90d_ma15, label='pred-90d-ma15')
    ax[1][0].plot(true_90d, label='true_90d')
    ax[1][0].set_title('pred Vs true return - 90d')
    ax[1][0].legend()

    ax[1][1].plot(pred_182d_ma30, label='pred-182d-ma30')
    ax[1][1].plot(true_182d, label='true_182d')
    ax[1][1].set_title('pred Vs true return - 192d')
    ax[1][1].legend()

    plt.savefig('eda/plots/PredVsTrueReturn.png')
    plt.show()


def plot_quantile_return_density(lag_res, feature):
    fig, ax = plt.subplots(2,2)
    coordinates = [(i, j) for i in range(2) for j in range(2)]
    k = 0

    for period in ['030', '060', '090', '182']:
        return_period = f'return_{period}d'
        df = lag_res.loc[feature, return_period]
        i, j = coordinates[k]
        sns.kdeplot(
                data=df,
                ax=ax[i][j],
            ).set_title(f'Quantile-Return-Density - {feature, period}')
        k += 1

    plt.show()
    

