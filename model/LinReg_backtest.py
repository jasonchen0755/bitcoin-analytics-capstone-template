import logging
import pandas as pd
from pathlib import Path

# Import template components
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis

# Import dca model
from model.LinReg import (
    _prepare_dataset,
    _predict_return,
    compute_quantile_winrate,
    compute_quantile_layered_return,
    compute_signal,
    compute_weight
)

# Global variable to store precomputed features
_SIGNAL= None

def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Wrapper for Example 1 compute_window_weights.
    
    Adapts the specific Example 1 model function to the interface expected 
    by the template backtest engine.
    """
    global _SIGNAL
    
    # if _FEATURES_DF is None and _TARGET_DF is None:
    #     raise ValueError("Dataset not prepared. Call _prepare_dataset() first.")
        
    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    
    # For backtesting, current_date = end_date (all dates are in the past)
    current_date = end_date
    
    return compute_weight(_SIGNAL, start_date, end_date)

def main():
    global _SIGNAL
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.info("Starting Bitcoin DCA Strategy Analysis - DCA (Technical & OnChain)")
    
    # 1. Load Data
    btc_df = load_data().loc[pd.date_range('2018-01-01', '2025-12-31')]
    
    # 2. Precompute Features (using Example 1 logic)
    logging.info("Preparing dataset, including Onchain data and yf-finance data)...")
    features, targets = _prepare_dataset()
    start_date, end_date = features.index.min(), features.index.max()
    prediction = _predict_return(features, targets)
    lag_res, quantiles = compute_quantile_winrate(features, targets)
    quantile_layered_return = compute_quantile_layered_return(features, lag_res, quantiles)
    df = pd.concat([prediction, quantile_layered_return], axis=1).reset_index().set_index('time').sort_index() # type: ignore
    _SIGNAL = compute_signal(df, '2018-01-01', '2025-12-31')
    
    # 3. Define Output Directory
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output_LinReg"
    
    # 4. Run Analysis (reusing Template engine)
    run_full_analysis(
        btc_df=btc_df,
        features_df = _SIGNAL,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="DCA (Linear Regression)"
    )

if __name__ == "__main__":
    main()
