import random  # Import the random module for generating random values.
import yfinance as yf  # Import yfinance for downloading stock data.
import pandas as pd  # Import pandas for data manipulation.
import matplotlib.pyplot as plt  # Import matplotlib for plotting.
from datetime import date, timedelta  # Import date and timedelta from datetime for date operations.
import numpy as np  # Import numpy for mathematical operations.

# Function to initialize variables
def VarInit():
    B_S_Factor = random.uniform(0.2, 4.5)  # Initialize B_S_Factor with a random value between 0.2 and 4.5.
    Transaction_Amount = random.uniform(0, 5)  # Initialize Transaction_Amount with a random value between 0 and 5.
    N_Value = random.uniform(3, 365)  # Initialize N_Value with a random value between 3 and 365.
    Cash = 100000  # Initialize Cash with a value of 100,000.
    Shares = 0  # Initialize Shares with a value of 0.
    return {"B_S_Factor": B_S_Factor, "Transaction_Amount": Transaction_Amount, "N_Value": N_Value, "Cash": Cash, "Shares": Shares}

# Function to download stock price data
def PriceDownload(ticker, starttime):
    try:
        Model_Data = yf.download(ticker, start=date.today() - timedelta(days=starttime), end=date.today(), progress=False)  # Download stock data using yfinance.
    except KeyError:
        return False
    if Model_Data.empty:
        return False
    Model_Data = Model_Data["Open"].dropna().to_numpy()  # Extract "Open" prices and convert them to a numpy array.
    return Model_Data

# Functions for price indicators
def price_up_indicator(upCheck, curPrice):
    return curPrice >= upCheck

def price_down_indicator(lowCheck, curPrice):
    return curPrice <= lowCheck

# Function to split data into training and simulation sets
def split_data(N_Value, data):
    train = data[:int(N_Value)]  # Split the data into training portion.
    simu_data = data[int(N_Value):]  # Split the data into simulation portion.
    mean = np.mean(train)  # Calculate the mean of the training data.
    STDV = np.std(data)  # Calculate the standard deviation of the entire data.
    return {"simu_data": simu_data, "mean": mean, "STDV": STDV}

# Buy and sell functions
def buy(price, cash, shares, Transaction_Amount=1):
    if cash >= price * Transaction_Amount:
        cash -= price * Transaction_Amount
        shares += Transaction_Amount
    return cash, shares

def sell(price, cash, shares, Transaction_Amount=1):
    if shares >= Transaction_Amount:
        cash += Transaction_Amount * price
        shares -= Transaction_Amount
    return cash, shares

# MAIN
Outout = {"Ticker:": "", "Cash": "", "Shares": 0, "Price": 0, "B_S_Factor": 0, "N_Value": 0}  # Initialize the output dictionary.
Tickers = ["GOOGL", "ADBE", "CRM", "TXN", "AMD", "ADSK", "PANW", "MU", "KLAC", "QCOM", "META", "MSFT", "AAPL", "ADI", "NXPI", "AVGO", "CTSH", "INTU", "ANSS", "AMAT", "DDOG", "NVDA", "WDAY", "FTNT"]  # List of stock tickers to analyze.
for ticker in Tickers:
    variables = VarInit()  # Initialize variables for each stock.
    raw_data = PriceDownload(ticker, 700)  # Download historical price data for the stock (adjust the number of days as needed).
    
    split_data_result = split_data(int(variables['N_Value']), raw_data)  # Split the data into training and simulation sets.
    
    cash = variables["Cash"]  # Initialize cash with the specified value.
    shares = variables["Shares"]  # Initialize shares with 0.
    last_trade = ""  # Initialize the last_trade variable.
    
    for price in split_data_result["simu_data"]:
        up_check = split_data_result["mean"] + variables["B_S_Factor"] * split_data_result["STDV"]  # Calculate the upper threshold for trading.
        low_check = split_data_result["mean"] - variables["B_S_Factor"] * split_data_result["STDV"]  # Calculate the lower threshold for trading.
    
        # Implement trading logic based on price movement and last trade action.
        if last_trade == "sold" and price <= split_data_result["mean"]:
            cash, shares = buy(price, cash, shares, int(variables['Transaction_Amount']))
            last_trade = ""
        if last_trade == "bought" and price >= split_data_result["mean"]:
            cash, shares = sell(price, cash, shares, int(variables['Transaction_Amount']))
            last_trade = ""
        if price_up_indicator(up_check, price):
            cash, shares = sell(price, cash, shares, int(variables['Transaction_Amount']))
        if price_down_indicator(low_check, price):
            cash, shares = buy(price, cash, shares, int(variables['Transaction_Amount']))

    print(f'Portfolio value:{cash + shares * price}')  # Calculate and print the portfolio value.
