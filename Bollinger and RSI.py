import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
df = pd.read_csv("EURUSD_Candlestick_1_D_BID_04.05.2003-21.01.2023.csv")

df["Gmt time"]=df["Gmt time"].str.replace(".000","")
df['Gmt time']=pd.to_datetime(df['Gmt time'],format='%d.%m.%Y %H:%M:%S')
df.set_index("Gmt time", inplace=True)
df=df[df.High!=df.Low]

df["VWAP"] = ta.vwap(df.High,df.Low,df.Close,df.Volume)
df["RSI"] = ta.rsi(df.Close, length=16)
my_bbands = ta.bbands(df.Close, length=14, std=2)
df = df.join(my_bbands)

VWAPsignal = [0] * len(df)
backcandles = 15
for row in range(backcandles,len(df)):
    upt = 1
    dnt = 1
    for i in range(row-backcandles, row+1):
        if max(df.Open[i], df.Close[i]) >= df.VWAP[i]:
            dnt = 0
        if min(df.Open[i], df.Close[i]) <= df.VWAP[i]:
            upt = 0
        if upt == 1 and dnt == 1:
            VWAPsignal[row] = 3
        elif upt == 1:
            VWAPsignal[row] = 2
        elif dnt == 1:
            VWAPsignal[row] = 1

df["VWAPsignal"] = VWAPsignal

def TotalSinal(i):
    if (df.VWAPsignal[i] == 2 and
        df.Close[i] <= df["BBL_14_2.0"][i] and
        df.RSI[i] < 45):
            return 2
    if (df.VWAPsignal[i] == 1 and
        df.Close[i] >= df["BBU_14_2.0"][i] and
        df.RSI[i] > 55):
            return 2

TotSignal = [0]*len(df)

for row in range(backcandles, len(df)):
     TotSignal[row] = TotalSinal(row)

df["TotalSignal"] = TotSignal

def pointposbreak(x):
     if x['TotalSignal'] == 1:
          return x["High"]+1e-4
     if x['TotalSignal'] == 2:
          return x["Low"]-1e-4
     else:
        return np.nan

df["pointposbreak"] = df.apply(lambda row: pointposbreak(row),axis=1)

st=10400
dfpl = df[:360]
dfpl.reset_index(inplace=True)
fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close']),
                go.Scatter(x=dfpl.index, y=dfpl.VWAP, 
                           line=dict(color='blue', width=1), 
                           name="VWAP"), 
                go.Scatter(x=dfpl.index, y=dfpl['BBL_14_2.0'], 
                           line=dict(color='green', width=1), 
                           name="BBL"),
                go.Scatter(x=dfpl.index, y=dfpl['BBU_14_2.0'], 
                           line=dict(color='green', width=1), 
                           name="BBU")])

fig.add_scatter(x=dfpl.index, y=dfpl['pointposbreak'], mode="markers",
                marker=dict(size=10, color="MediumPurple"),
                name="Signal")
fig.show()
