import alpaca_trade_api as tradeapi
import plotly.graph_objects as go
from plotly.subplots import make_subplots

key = ""
secret = ""
api = tradeapi.REST(key, secret, "https://paper-api.alpaca.markets")

api.get_account()

df = api.get_bars(symbol="AAPL", start="2023-10-27", timeframe="5min").df



# To drop specific columns, you should use the df.drop() method as follows:
columns_to_drop = ["trade_count", "volume", "vwap"]
df = df.drop(columns=columns_to_drop)

# Candlestick plot
fig = go.Figure(data=[go.Candlestick(x=df.index,
                                     open=df['open'],
                                     high=df["high"],
                                     low=df['low'],
                                     close=df["close"]
                                     )])

fig.update_layout(title_text="Candlestick Chart")

def add_rs_val(price_diff, rs_dict):
    if price_diff > 0:
        rs_dict["hi"].append(price_diff)
        rs_dict["lo"].append(0)  # If it's a gain, add 0 to losses
    else:
        rs_dict["hi"].append(0)  # If it's a loss, add 0 to gains
        rs_dict["lo"].append(abs(price_diff))
    return rs_dict

def add_rsi_val(rs_dict, rsi_arr):
    average_gain = sum(rs_dict["hi"]) / 14
    average_loss = sum(rs_dict["lo"]) / 14
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_arr.append(rsi)
    return rsi_arr

def add_stoch_val(rsi_arr, stoch_rsi_arr):
    min_rsi = min(rsi_arr[-14:])  # Get the most recent 14 RSI values
    max_rsi = max(rsi_arr[-14:])
    cur_rsi = rsi_arr[-1]
    stoch_rsi = (cur_rsi - min_rsi) / (max_rsi - min_rsi)
    stoch_rsi_arr.append(stoch_rsi)
    return stoch_rsi_arr

rs_dict = {"hi": [], "lo": []}
rsi_arr = []
stoch_rsi_arr = []

# We should calculate RS 14 times before calculating RSI
for i in range(14):
    rs_dict = add_rs_val(df["open"][i] - df["close"][i], rs_dict)

# We should then calculate RSI 10 times before calculating Stochastic RSI
for i in range(14, 24):
    rsi_arr = add_rsi_val(rs_dict, rsi_arr)
    rs_dict = add_rs_val(df["open"][i] - df["close"][i], rs_dict)

# Now continuously calculate Stochastic RSI
for i in range(24, len(df)):
    stoch_rsi_arr = add_stoch_val(rsi_arr, stoch_rsi_arr)
    rs_dict = add_rs_val(df["open"][i] - df["close"][i], rs_dict)
    rsi_arr = add_rsi_val(rs_dict, rsi_arr)

# Now create a separate figure for Stochastic RSI
stoch_rsi_fig = go.Figure()
stoch_rsi_fig.add_trace(go.Scatter(x=df.index[24:], y=stoch_rsi_arr, mode='lines', name='Stochastic RSI'))

stoch_rsi_fig.update_layout(title_text="Stochastic RSI")

# Show the candlestick figure and the Stochastic RSI figure separately
fig.show()
stoch_rsi_fig.show()
