"""This code is optimizing a moving average buy-sell strategy alongside a Fibonacci stop loss. The stop loss is set below the "golden ratio" and it is also optimizing the length of the fib window.
This turned out to be a pretty good strategy. It yielded a 13% average (the average of the optimized models for 19 stocks), with an average of 32% exposure time, whereas the buy and hold average yielded -5%
"""


import pandas as pd
from backtesting import Backtest, Strategy
import pandas_ta as ta
import numpy as np
import warnings 
from scipy import stats


warnings.filterwarnings("ignore")
max_tries = 50

def optimize_ratio(stats):
    if stats["# Trades"] <= 10: return 0
    return (stats["Equity Final [$]"]**2/(stats["Exposure Time [%]"]))

def pass_func(x):
    return x

def fib_nums(Close,length):
    fib_highs = np.zeros_like(Close) 
    fib_lows  = np.zeros_like(Close) 

    for i in range(length,len(Close)):
        fib_highs[i] = max(Close[i-length:i+1])
        fib_lows[i] = max(Close[i-length:i+1])

    return fib_highs,fib_lows

def rate_of_change(Close, length):
    roc = np.zeros_like(Close) 

    for i in range(length, len(Close)):
        sum_roc = 0
        for j in range(i - length, i + 1):
            try:
                cur_roc = (Close[j] - Close[j - 1]) / Close[j - 1]
                sum_roc += cur_roc
            except:
                pass
        
        roc[i] = sum_roc / length

    return roc



class MA_strat(Strategy):
    ma_length = 20 
    roc_length = 3
    fib_length = 10

    def init(self): 
        ma = ta.ma("sma", self.data.df.Close, length=self.ma_length)
        self.ma = self.I(pass_func, ma)
        high, lows = fib_nums(self.data.df.Close,self.fib_length)

        self.fib_high = self.I(pass_func,high)
        self.fib_low  = self.I(pass_func,lows)
        self.roc = self.I(rate_of_change, self.data.df.Close.values, self.roc_length)

    def next(self):
        fib_high = self.fib_high
        fib_low = self.fib_low
        ma_val = float(self.ma[-1])
        close = float(self.data.Close[-1])
        roc = float(self.roc[-1])
        
        if fib_high >= close:
            tp = close*1.15
        else: tp = fib_high

        if fib_low <= close:
            sl = fib_low + (fib_high-fib_low)*.38 #40% above fib low
        else: sl = close*.9

        if ma_val >= close and roc > 0:
            self.position.close()
        elif ma_val <= close and roc < 0:
            self.buy(sl=sl,tp=tp)

#Set up date
df = pd.read_csv("data.csv")
df["Date"] = pd.to_datetime(df["Date"])  # Correct date format
tickers = df["Ticker"].unique()
df = df.set_index(["Ticker", "Date"]).sort_index()  # Set MultiIndex with Ticker first

output = pd.DataFrame(columns = ["Ticker","ma_length","roc_length","Return","exposure time","buy hold"])
# Test algo

for i,ticker in enumerate(tickers):
    cur_df = df.xs(ticker, level="Ticker")
    bt = Backtest(cur_df, MA_strat, cash=10_000, commission=.002)

    #Pull an clean optimized data
    stats = bt.optimize(ma_length=range(10,75,2),
                        roc_length=(range(2,15,1)),
                        fib_length = range(5,31,2),
                        maximize=optimize_ratio,
                        max_tries=40)
    strat = (stats["_strategy"])
    ROI = (stats["Return [%]"])
    buy_hold = stats["Buy & Hold Return [%]"]
    expose = (stats["Exposure Time [%]"])
    my_str = str(strat)
    my_str = my_str.split("(")[1].split(",")
    ma_length = my_str[0].split("=")[1]
    roc_length = my_str[1].split("=")[1].strip(")")

    #Put data on dataframe
    output.at[i,"Ticker"] = ticker
    output.at[i,"ma_length"] = ma_length
    output.at[i,"roc_length"] = roc_length
    output.at[i,"Return"] = ROI
    output.at[i,"exposure time"] = expose
    output.at[i,"buy hold"] = buy_hold

output.to_csv("MA_results.csv",index=False)




