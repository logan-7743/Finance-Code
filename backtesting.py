from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG
import pandas_ta as ta
import pandas as pd
GOOG.dropna(inplace=True)

class RsiOscillator(Strategy):
    upper_bound = 60
    lower_bound = 30
    rsi_period = 14

    def init(self): 
        # Remove NaN values from the 'Close' column


        # Calculate RSI and store it in a variable
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), length=self.rsi_period)

    def next(self):
        # Access the RSI values directly from the variable
        rsi_value = self.rsi[-1]

        if rsi_value >= self.upper_bound:
            self.position.close()
        elif rsi_value <= self.lower_bound:
            self.buy()

bt = Backtest(GOOG, RsiOscillator, cash=10_000, commission=.002)
stats = bt.optimize(upper_bound = range(55,95,5),
                    lower_bound = range(5,50,5),
                    rsi_period = range(7,30,2),
                    maximize= "Return [%]"
)
print(stats)
print(bt.plot())
