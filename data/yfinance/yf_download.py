import yfinance as yf
import pandas as pd


def yf_download():
    tickers = ["BTC-USD", "GC=F", "DX-Y.NYB"]

    all_res = {}
    for ticker in tickers:
        data = yf.download(
                        ticker,
                        start="2010-07-18",
                        interval="1d",
                        progress=False,
                        threads=False
                    )
        pd.DataFrame(data).to_csv(f"data/yfinance/{ticker.replace('=','_')}.csv")
        all_res[ticker] = data

    return all_res