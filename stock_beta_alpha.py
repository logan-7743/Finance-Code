"""This code calculates a stock beta and alpha. It is using daily data for the past two years. The purpose for this calcualtion (as oppsed to 5 year beta with monthly data),
is becasue I want the beta to refelct the data that I am going to be using for stock annaylis, namley, daily data, for the past two years. The code is pretty simple,
It calcualtes the rate of change ont the two then does a regression on them. I am just doing this for the list of tickers that I am using.
"""


import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

def beta_alpah(stock_ticker):
    market_index_ticker = '^GSPC'  # S&P 500

    # Fetch historical price data
    stock_data = yf.download(stock_ticker, start='2021-12-27', end='2023-12-27')['Adj Close']
    market_data = yf.download(market_index_ticker, start='2022-01-01', end='2022-12-31')['Adj Close']

    # Combine data into a DataFrame
    data = pd.concat([stock_data, market_data], axis=1)
    data.columns = [stock_ticker, market_index_ticker]

    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Calculate beta using linear regression (Covariance / Variance)
    beta, alpha = stats.linregress(returns[market_index_ticker], returns[stock_ticker])[:2]
    return beta, alpha
df = pd.read_csv("merged_data.csv")
tickers = df["Ticker"].unique()

b_a_dict = {}
for ticker in tickers:
    b,a  = beta_alpah(ticker)
    b_a_dict[ticker] = (b,a)

df["beta"] = [0]*len(df)
df["alpha"] = [0]*len(df)
for i in range(len(df)):
    ticker = df.at[i,"Ticker"]
    df.at[i,"beta"] = b_a_dict[ticker][0]
    df.at[i,"alpha"] = b_a_dict[ticker][1]
df.to_csv("merge_data_w_beta_alpha.csv")
