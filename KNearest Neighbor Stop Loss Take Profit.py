# from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
# import plotly.graph_objects as go
# from datetime import datetime
# import matplotlib as plt
import matplotlib.pyplot as plt
# from matplotlib import pyplot
import numpy as np
import pandas_ta as ta
from scipy.stats import linregress
df = pd.read_csv("EURUSD_Candlestick_1_D_BID_04.05.2003-21.01.2023.csv")

#Check if any zero volumes are available
indexZeros = df[ df['Volume'] == 0 ].index

df.drop(indexZeros , inplace=True)
df.loc[(df["Volume"] == 0 )]

df["ATR"] = df.ta.atr(length=20)
df["RSI"] = df.ta.rsi()
df["Average"] = df.ta.midprice(length=1)
df["MA40"] = df.ta.sma(length=40)
df["MA80"] = df.ta.sma(length=80)
df["MA160"] = df.ta.sma(length=160)

def get_slope(array):
    y = np.array(array)
    x = np.arange(len(array))
    slope,_,_,_,_ = linregress(x,y)
    return slope

backdollingN = 6

df["slopeMA40"]    = df["MA40"].rolling(window=backdollingN).apply(get_slope, raw = True)
df["slopeMA80"]    = df["MA80"].rolling(window=backdollingN).apply(get_slope, raw = True)
df["slopeMA160"]   = df["MA160"].rolling(window=backdollingN).apply(get_slope, raw = True)
df['AverageSlope'] = df['Average'].rolling(window=backdollingN).apply(get_slope, raw=True)
df['RSISlope']     = df['RSI'].rolling(window=backdollingN).apply(get_slope, raw=True)

#Target flexible way
pipdiff = 500*1e-5 #for TP
SLTPRatio = 2 #pipdiff/Ratio gives SL

def mytarget(barsupfront, df1):
    length = len(df1)
    high = list(df1['High'])
    low = list(df1['Low'])
    close = list(df1['Close'])
    open = list(df1['Open'])
    trendcat = [None] * length
    
    for line in range (0,length-barsupfront-2):
        valueOpenLow = 0
        valueOpenHigh = 0
        for i in range(1,barsupfront+2):
            value1 = open[line+1]-low[line+i]
            value2 = open[line+1]-high[line+i]
            valueOpenLow = max(value1, valueOpenLow)
            valueOpenHigh = min(value2, valueOpenHigh)

            if ( (valueOpenLow >= pipdiff) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
                trendcat[line] = 1 #-1 downtrend
                break
            elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= pipdiff) ):
                trendcat[line] = 2 # uptrend
                break
            else:
                trendcat[line] = 0 # no clear trend
            
    return trendcat

df["mytarget"] = mytarget(16,df)
# fig = plt.figure(figsize = (15,20))
# ax = fig.gca()
# print(ax)
# df_model= df[['Volume', 'ATR', 'RSI', 'Average', 'MA40', 'MA80', 'MA160', 'slopeMA40', 'slopeMA80', 'slopeMA160', 'AverageSlope', 'RSISlope', 'mytarget']] 
# df_model.hist(ax = ax)
# plt.show()

# df_up=df.RSI[ df['mytarget'] == 2 ]
# df_down=df.RSI[ df['mytarget'] == 1 ]
# df_unclear=df.RSI[ df['mytarget'] == 0 ]
# pyplot.hist(df_unclear, bins=100, label='unclear')
# pyplot.hist(df_down, bins=100, label='down')
# pyplot.hist(df_up, bins=100, label='up')

# pyplot.legend(loc='upper right')
# pyplot.show()

df_model= df[['Volume', 'ATR', 'RSI', 'Average', 'MA40', 'MA80', 'MA160', 'slopeMA40', 'slopeMA80', 'slopeMA160', 'AverageSlope', 'RSISlope', 'mytarget']] 
df_model=df_model.dropna()

attributes=['ATR', 'RSI', 'Average', 'MA40', 'MA80', 'MA160', 'slopeMA40', 'slopeMA80', 'slopeMA160', 'AverageSlope', 'RSISlope']
X = df_model[attributes]
y = df_model["mytarget"]

train_index = int(.8*len(X))
X_train, X_test = X[:train_index], X[train_index:]    
y_train, y_test = y[:train_index], y[train_index:]

model = KNeighborsClassifier(n_neighbors=250)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Accuracy train: %.2f%%" % (accuracy_train * 100.0))
print("Accuracy test: %.2f%%" % (accuracy_test * 100.0))

print(df_model['mytarget'].value_counts()*100/df_model['mytarget'].count())
