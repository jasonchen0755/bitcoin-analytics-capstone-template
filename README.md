# Predictive Boosting DCA

The goal of this project is to design a dynamic Dollar-Cost-Average strategy, which adopts long-only method, helping accumulate as much bitcoins as it could, and significantly outperforming passive uniform DCA strategy.
While the ultimate purpose of analytics is to give predictive signals, helping not only in accumulating, but also in trading transactions, it is temping to develop a model which could ouput credible future price/return forcasts. 

In this project, I have developed two models to predict future return. One is a simple linear regression method, the other uses Mamba achietecture as basic blocks of a 4-layers deep learning model. Then the algorithms use predicted results either as signal, or as booster of uniform weights. The final model combined metrics used in the two models. Specifically, the signals first boosted by Mamba predictions, then amplified by technical and on-chain metrics which are helpful to detect paradigm shift. The final model wins **71.84%** of the test windows over uniform strategy.

**NOTE** 

Extra packages are needed to run backtests (see requirements.txt). The mamba-ssm installation is tedious and time consuming. Screen shots of backtests were saved in 'model/output_LinReg', 'model/output_mamba', and 'model/output_final' in case when readers are unwilling to install them.

---

## Backtest results

1.  **Linear Regression method**
    ```bash
    python -m model.LinReg_backtest
    ```
    <img src='model/output_LinReg/LinReg_backtest_screenshot.png' width='600'>

    This simplest predictive method gets **50.29%** win rate, which is not very different from uniform strategy. All backtest ouputs were saved in folder 'model/output_LinReg'.

2.  **Mamba based deep learning method**
    ```bash
    python -m model.mamba_backtest
    ```

    <img src='model/output_mamba/mamba_backtest_screenshot.png' width='600'>

    This method outputs far more reliable signals and gets **66.45%** win rate. To further improve it, finer retraining interval and/or finding other informative signals could be viable directions. All backtest ouputs were saved in folder 'model/output_mamba'.

    Mamaba achietecture is good on time series tasks. It is a State-Space model introduced by Albert Gu et.al. in 2024. [MAMBA SSM Achitecture] (https://github.com/state-spaces/mamba)

3.  **Final Combination**
    ```bash
    python -m model.final_model_backtest
    ```

    <img src='model/output_final/performance_comparison.svg' width=800>

    The final model starts from signals learned by mamba method, then introduces Mayer-multiple as metrics to identify paradigm shift. When paradigm seems shifted, the model amplifies signals if RSI and HarshRate fell in extreme quantiles. The final model prevails Uniform strategy with **71.84%** win rate.

    ```python
    # (1) Rolling sd: Here we use 3 days rolling standard deviation of 
    #       30 days return, which may capture short term volatility. 
    res_std_030d = res['return_030d'].rolling(3).std().bfill().shift(1)

    # (2) Mayer multiple: It is the ratio of current Bitcoin price by 
    #       its 200-day moving average, represents long term reversion force.
    sma_200 = close.rolling(window=200).mean()
    Mayer_multiple = close / sma_200  

    # (3) RSI: The Relative Strength Index is constructed on daily gains 
    #   and losses. It is a momentum oscillator that measures the speed 
    #   and magnitude of price movements to identify overbought or oversold.
    delta = close.diff(1)
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    rsi = gain / loss
    RSI = 100 - (100 / (1+rsi))
    
    # (4) HashRate: we care about the distance of ma7 to ma30. 
    ma7 = HashRate.rolling(7, 3).mean()
    ma30 = HashRate.rolling(30, 15).mean()
    HashRate_ma7_ma30 = (ma7/ma30 - 1).clip(-1, 1)
    ```

## Extra packages & Function Dependency

1.  **For EDA and Linear Regress method**
    
    yfinance==1.1.0

    ipykernel==7.1.0

    statsmodels==0.14.6

    scikit-learn==1.8.0

    hmmlearn==0.3.3

2.  **For mamba method**
    
    torch==2.11.0

    mamba_ssm==2.3.1

    Please note 'mamba_ssm' installation needs Linux system and cuda available.

3.  **Functions Dependency**

    ```text
    Linear Regression method:
        model/LinReg_backtest.py
        model/LinReg.py
        template/prelude_template.py
        templat/backtest_template.py
    Mamba method:
        model/mamba_backtest.py
        model/mamba.py
        model/prelude_template_mamba.py     # copied from template/prelude_template.py
        model/backtest_template_mamba.py    # copied from template/backtest_template.py
    Final model:
        model/final_model_backtest.py
        model/final_model_prelude_template.py   # copied from model/prelude_template_mamba.py
        model/final_model_backtest_template.py  # copied from model/backtest_template_mamba.py
    ```

## Mamba Method Explaination

1.  **Model Achietecture**
    ```text
    ├── Input projection layer           # 16 -> 256 dims
    ├── Basic blocks * 4                 # 4 layers of repeated basic blocks
    │   ├── Mamba                        # from mamba_ssm
    │   ├── Dropout                      # dropout=0.1
    │   └── LayerNorm                    # LayerNorm is helpful while BatchNorm is not    
    ├── Average Pooling                  # lambda x: x.mean(dim=1)
    ├── Head                             # output layer
    │   ├── Linear                       # 256 -> 256
    │   ├── Relu                         # regularization
    │   ├── Dropout                      # dropout=0.1
    └── └── Linear                       # 256 -> 4   
    ```

2.  **Features Slection & Target Horizons**

    ```python
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
    HORIZONS = ['return_030d',
                'return_060d', 
                'return_090d', 
                'return_120d']
    ```
    Technical features are computed on yahoo finance dataset, which could be downloaded by 'data/yfinance/yf_download.py'. OnChain features are computed on Coinmetrics dataset, and Poly features on polymarket dataset. Targets are future returns if buy at today's price, in 4 horizons.

3.  **Training Data Structure & Models Periodical Coverage**
    
    (1) Organized x_train, x_val, and x_test:

    For the time series dataset task, data structure should be in a info-leakage-preventing manner, for example, the data feed into model name 'model/checkpoint/model_2018-12-31.pt' should be: (see function 'load_data' in model/mamba.py)
    ```python
    N = len(pd.date_range('2010-07-18','2018-12-31'))
    n_test = max(int(N * test_ratio), 365)
    n_val = int(N * val_ratio)
    n_train = N - n_val - n_test

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train: n_train + n_val - 120], Y[n_train: n_train + n_val - 120]
    X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]
    ```
    
    Please note that the last 365 steps data never involved in training nor in model selection. And the 120 steps before test data are masked as well since my longest target horizon is 120 days return. 

    (2) For every test window [start_date: end_date], call relevant pretrained model to predict signal:

    ```text
    end_date < 2018-12-31 --> model_2018-12-31.pt
    end_date < 2019-06-30 --> model_2019-06-30.pt
    end_date < 2019-12-31 --> model_2019-12-31.pt
    ...
    ```
    (3) Signals are formatted as  {start_date:np.ndarray of 365 signals}, for instance:
    ```text
    {'2018-01-01': [365 signals]}
    {'2018-01-02': [365 signals]}
    ...
    ```
    (4) Since the signals are computed in a rolling manner, below backtest functions are revised accordingly:
    ```text
    backtest_template_mamba.py --> copied from backtest_template.py
            (run_full_analysis)
    prelude_template_mamba.py --> copied from prelude_template.py
            (check_strategy_submission_ready, backtest_dynamic_dca, compute_cycle_spd)
    ```

4.  **Prediction Quality**

    ```bash
    python -m model.template_mamba
    ```

    <img src='model/output_mamba/best_model_120d.png' width='800'>

    | Metrics |  30d   |  60d  |  90d  |  120d |
    |---------|--------|-------|-------|-------|
    | MSE | 3.2640838e-03 | 2.4250933e-04 | 1.5079099e-04 | 9.9083205e-05 |
    | MAE | 2.4871876 | 2.7896104 | 1.9521879 | 1.036711 |
    | Direction Accuracy | 0.57798165 | 0.72247706 | 0.71330275 | 0.90825688 |

    The model predicts future returns based on historical features with sequence length of 128 time steps. So it is expected that the 120d horizon reports best metrics. And this design is intended since we are adopting long-only philosophy. 

    In back-testing stage, we choose predictions on 120d horizon as signal. We first set a threshold of |signal| > 0.05, then tune it. It is regarded as credible signal and be added on uniform weight as booster or negator after multiplied by a constant. Statiscally, uniform DCA strategy copes well with volatile assets. Predictive signals alone could not beat uniform method. 

## Combination and Improvements

1.  **Paradigm Shift Detection**

    Below command runs the custom-backtest function which cares about the history of Mamba method competing with Uniform strategy. The vertical dash line represent the time we periodically switch pre-trained models. Green bars indicate our strategy is wining, red bars represent uniform strategy is beating us, and some sparse orange dots alone horizontal zero line are ties.

    ```bash
    python -c 'from model.mamba_backtest import custom_backtest; custom_backtest()'
    ```

    <img src='model/output_mamba/Mamba_Vs_Unif_Battle_history_plot.png' width=800>

    It is observed that the win-loss-territory changes are mostly near the time when we periodically switched models. In almost all test windows which start from one day in year 2020, the proposed strategy failed. The reason should be investigated carefully. Two possibilities worth try: (1) Train models more frequently, say, per month. (2) Introduce other boosters or negators.

    And it is interesting to observe that this kind of paradigm shifts are quite similar to those captured by Hidden Markov Model. The model performs good in state 0 and struggles in state 1:

    ```bash
    python -c 'from model.utils import hmm_state; hmm_state()'
    ```

    <img src='model/output_mamba/Hidden_Markov_State_Recognision_plot.png' width=800>


2.  **Quantile-Layered returns of select features**

    Some features, such as ['HashRate_ma7_ma30', 'RSI', 'mvrv_zscore', 'price_ma7_Ma30'], when fell into extreme quantiles (below 0.1 or above 0.9), are strong indicators of future return.

    ```bash
    python -c 'from model.utils import plot_quantile_return_density; 
                from model.LinReg import _prepare_dataset, compute_quantile_winrate; 
                x, y = _prepare_dataset(); 
                lag_res, quantiles = compute_quantile_winrate(x, y); 
                plot_quantile_return_density(lag_res, "HashRate_ma7_ma30")'
    ```

    <img src='model/output_mamba/HashRate_Quantile_Return_Density_plot.png' width=800>

    These features could be used to construct credible booster/negator signals when they fell into extreme quantiles. And hopefully doing so will improve strategy performance further.

3.  **Improvements**

    Two metrics, **standard deviation of 30 days return**, along with **Mayer-multiple**, are helpful to detect paradigm shift. And two other metric, **HashRate_ma7_ma30** and **RSI**, are used to amplify signals when the value of them fell in below 10% or above 90% quantiles. This simple maniplation raises win rate from **66.45%** to **71.84%**, and gets meaningful surplus over uniform strategy.

    <img src='model/output_final/cumulative_performance.svg' width=800>

## END