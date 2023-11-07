import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import matplotlib as plt
df = pd.read_csv("EURUSD_Candlestick_1_D_BID_04.05.2003-21.01.2023.csv")

# drop days with no volume
df = df[df["Volume"]!=0]
df.reset_index(drop=True, inplace=True)
df.isna().sum()

def support(df1, l, n1, n2): #n1 n2 before and after L
    for i in range(l-n1+1,l+1): #check left side. Check if all previous stocks are lessee then each predescceor
        if(df1.Low[i]>df1.Low[i-1]): #Check right side
            return 0
    for i in range(l+1,l+n2+1): #check if all stick are less then succesor
        if(df1.Low[i]<df1.Low[i-1]):
            return 0
    return 1

def resistance(df1, l, n1, n2):
    for i in range(l-n1+1,l+1):
        if df1.High[i]<df1.High[i-1]:
            return 0
        
    for i in range(l+1,l+n2+1):
        if df1.High[i]>df1.High[i-1]:
            return 0
    return 2

sr = []
n1=3
n2=2
for row in range(n1,205):
    if support(df,row,n1,n2):
        sr.append((row,df.Low[row],1))
    if resistance(df,row,n1,n2):
        sr.append((row,df.High[row],2))

s = 0
e = 200
dfpl = df[s:e]

fig= go.Figure(data=[go.Candlestick(x=dfpl.index,
                open = dfpl["Open"],
                high = dfpl["High"],
                low = dfpl["Low"],
                close = dfpl["Close"])])


plotlist1 = [x[1] for x in sr if x[2]==1]
plotlist2 = [x[1] for x in sr if x[2]==2]

for i in range(1,len(plotlist2)):
    if(i>=len(plotlist2)):
        break
    if abs(plotlist2[i]-plotlist2[i-1])<=0.005:
        plotlist2.pop(i)


for i in range(1,len(plotlist1)):
    if(i>=len(plotlist1)):
        break
    if abs(plotlist1[i]-plotlist1[i-1])<=0.005:
        plotlist1.pop(i)

for i in range(len(plotlist1)):
    fig.add_shape(type="line", x0=s, y0=plotlist1[i],
                  x1=e,
                  y1=plotlist1[i])

for i in range(len(plotlist2)):
    fig.add_shape(type="line", x0=s, y0=plotlist2[i],
                  x1=e,
                  y1=plotlist2[i])
    
fig.show()
