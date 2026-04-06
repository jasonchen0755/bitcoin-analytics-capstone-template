import pandas as pd
import numpy as np

from model.utils import *
from sklearn.linear_model import LinearRegression as lg
import logging
import matplotlib.pyplot as plt


def prepare_features():
    """
    prepare and align features
    """
    logging.info('preparing polymarket features')
    poly, rate_down, rate_up = compute_polymarket_features()

    logging.info('preparing onchain features')
    onchain = compute_onchain_features()

    logging.info('preparing technical features')
    tech = compute_technical_metrics()

    return poly, onchain, tech

def reg(df:pd.DataFrame):
    """
    run regression of features on btc returns
    """
    targets = ['return_001d', 'return_007d', 'return_014d', 'return_030d', 'return_060d', 'return_090d', 'return_182d', 'return_365d']
    Ys = df[targets]
    Xs = df.drop(targets, axis=1)
    res = []

    xnames, ynames = Xs.columns, Ys.columns
    for xx in xnames:
        for yy in ynames:
            data = df[[xx, yy]]
            data = data[~((data[xx]==0) & (data[yy]==0))].dropna()
            x, y = data[xx].to_numpy().reshape(-1,1), data[yy].to_numpy()
            reg = lg()
            reg.fit(x, y)
            res.append({'y': yy, 'x': xx, 'intercept': reg.intercept_, 'coef': reg.coef_, 'R2': reg.score(x, y)})
    return res

def return_regression_on_feature():
    """
    run linear regression of features on different time-scal btc returns
    """
    poly, onchain, tech = prepare_features()
    logging.info('preparing btc return data')
    ret = compute_btc_returns().drop(['price'], axis=1)

    poly = pd.merge(poly, ret, how='left', left_index=True, right_index=True)
    onchain = pd.merge(onchain, ret, how='left', left_index=True, right_index=True)
    tech = pd.merge(tech, ret, how='left', left_index=True, right_index=True)

    logging.info('run linear regression of polymarket features on btc returns')
    res_poly = pd.DataFrame(reg(poly))
    logging.info('run linear regression of onchain features on btc returns')
    res_onchain = pd.DataFrame(reg(onchain))
    logging.info('run linear regression of technical features on btc returns')
    res_tech = pd.DataFrame(reg(tech))
    all_res = pd.concat([res_poly, res_onchain, res_tech]).sort_values('y')

    plot_significant_reg(poly, onchain, tech, all_res)
    plot_ma_reversion_with_return()
    all_res.to_csv('eda/return-feature-regression.csv')

    return all_res


def plot_significant_reg(poly, onchain, tech, all_res):
    """
    plot regression formula curve with R2>0.10 and corresponded targets
    """
    all_res = all_res[all_res['R2']>=0.10]

    n = len(all_res)
    j, i = int(np.floor(np.sqrt(n))), int(n // np.floor(np.sqrt(n)))
    if i < n / np.floor(np.sqrt(n)):
        i += 1
    coordinates = [[k, l] for k in range(i) for l in range(j)]
    fig, ax = plt.subplots(i, j, figsize=(j*10, i*10))
    k = 0

    for row in all_res.iterrows():
        record = pd.Series(row)[1]
        xname, yname = record['x'], record['y']
        intercept, coef, R2 = record['intercept'], record['coef'][0], record['R2']
            
        if xname in poly.columns:
            prediction = intercept + poly[xname] * coef
            actual = poly[yname]
        elif xname in onchain.columns:
            prediction = intercept + onchain[xname] * coef
            actual = onchain[yname]
        elif xname in tech.columns:
            prediction = intercept + tech[xname] * coef
            actual = tech[yname]   
        else:
            logging.warning(f'feature {xname} does not in any place')
            break
        try:
            i,j = coordinates[k]
            ax[i][j].plot(prediction.index, prediction.to_numpy(), label='predict')
            ax[i][j].plot(actual.index, actual.to_numpy(), label='actual')
            ax[i][j].set_title(f"{yname[-4:]} ~ {xname} / R2: {round(R2,2)}")
            ax[i][j].legend()
        except Exception as e:
            logging.warning(f'current row: {i}, current column: {j}, current subplot: {k}/{len(coordinates)}')
            print(e)

        k += 1  
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.2)
    plt.savefig('eda/plots/return-feature-regression-plot.png')
    plt.close()

def plot_ma_reversion_with_return():
    from .utils import compute_btc_returns
    ret = compute_btc_returns().loc['2012-01-01':]
    price = ret['price']

    reverse = lambda s1,s2: pd.Series((s1/s2-1).clip(-1,1).fillna(0), name=f"{s1.name}_{s2.name}")
    ma = lambda series,period:series.rolling(period).mean().fillna(0)
    ma7, ma14, ma30, ma60, ma90, ma182, ma365 = ma(price,7), ma(price,14), ma(price,30), ma(price,60), ma(price,90), ma(price,182), ma(price,365)

    fig, ax = plt.subplots(2,2)
    ax[0][0].plot(reverse(ma7,ma14), label='ma7_ma14')
    # ax[0][0].plot(reverse(ma7,ma30), label='ma7_ma30')
    ax[0][0].plot(reverse(ma7,ma60), label='ma7_ma60')
    ax[0][0].plot(reverse(price, ma(price,200)), label='price_vs_ma200')
    ax[0][0].plot(ret['return_030d'].clip(upper=3.0), label='return 30d')
    ax[0][0].axhline(y=0.0, color='red', linestyle='--')
    ax[0][0].legend()
    ax[0][0].set_title('reverse force - ma7/ret30')

    ax[0][1].plot(reverse(ma14,ma30), label='ma14_ma30')
    # ax[0][1].plot(reverse(ma14,ma60), label='ma14_ma60')
    ax[0][1].plot(reverse(ma14,ma90), label='ma14_ma90')
    ax[0][1].plot(reverse(price, ma(price,200)), label='price_vs_ma200')
    ax[0][1].plot(ret['return_060d'].clip(upper=3.0), label='return 60d')
    ax[0][1].axhline(y=0.0, color='red', linestyle='--')    
    ax[0][1].legend()
    ax[0][1].set_title('reverse force - ma14/ret60')

    ax[1][0].plot(reverse(ma30,ma60), label='ma30_ma60')
    # ax[1][0].plot(reverse(ma30,ma90), label='ma30_ma90')
    ax[1][0].plot(reverse(ma30,ma182), label='ma30_ma182')
    ax[1][0].plot(reverse(price, ma(price,200)), label='price_vs_ma200')
    ax[1][0].plot(ret['return_090d'].clip(upper=3.0), label='return 90d')
    ax[1][0].axhline(y=0.0, color='red', linestyle='--')     
    ax[1][0].legend()
    ax[1][0].set_title('reverse force - ma30/ret90')    

    ax[1][1].plot(reverse(ma60,ma90), label='ma60_ma90')
    # ax[1][1].plot(reverse(ma60,ma182), label='ma60_ma182')
    ax[1][1].plot(reverse(ma60,ma365), label='ma60_ma365')
    ax[1][1].plot(reverse(price, ma(price,200)), label='price_vs_ma200')
    ax[1][1].plot(ret['return_182d'].clip(upper=3.0), label='return 182d')   
    ax[1][1].axhline(y=0.0, color='red', linestyle='--')  
    ax[1][1].legend()
    ax[1][1].set_title('reverse force - ma60/ret182')  

    plt.savefig('eda/plots/ma_reversion.png')
    plt.close()

def main():
    return_regression_on_feature()

if __name__ == '__main__':
    main()