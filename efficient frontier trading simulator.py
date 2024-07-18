import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy.optimize import minimize
import warnings
import random
import time

# Download historical price data and calculate returns and covariance matrix
def get_data(stocks, start, end, interval="1d"):
    data = yf.download(stocks, start=start, end=end, interval=interval)
    if isinstance(stocks, list):
        data = data.dropna(axis=1, how='any')  # Drop columns with NaNs
    data = data["Close"]  # Select the "Close" column
    returns = np.log(data / data.shift(1)).dropna()
    return data, returns

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    p_return = np.sum(weights * mean_returns)
    p_stdv = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return p_return, p_stdv

# Function to calculate negative Sharpe ratio
def negative_sharpe(weights, mean_returns, cov_matrix, risk_free):
    p_return, p_stdv = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return**2 - risk_free) / p_stdv

# Function to maximize Sharpe ratio
def max_sharpe(mean_returns, cov_matrix, risk_free, current_weights):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free)
    
    # Constraint: Total weight can vary between 0 and 1
    constraints = {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)}  # Sum of weights <= 1 constraint
    
    # Adjust bounds to allow for weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    initial_guess = current_weights

    result = minimize(negative_sharpe, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result

# Function to perform portfolio simulation
def portfolio_simulation(df, sp500, start, end, lookback_days, interval, risk_free=.01):
    df_filtered    = df.loc[start:end].reset_index(drop=True)
    sp500_filtered = sp500.loc[start:end].reset_index(drop=True)
    
    initial_investment      = 100000  # Initial investment amount for all portfolios
    portfolio_values        = [initial_investment]  # Starting portfolio value for dynamic portfolio
    static_portfolio_values = [initial_investment]  # Starting value for static portfolio
    sp500_portfolio_values  = [initial_investment]  # Starting value for S&P 500 portfolio

    old_weights    = np.array([1 / len(df.columns)] * len(df.columns))
    static_weights = old_weights.copy()

    covariances = []  # To store covariance matrices

    for i in range(lookback_days, len(df_filtered)):
        try:
            # Grab data
            lookback_data    = df_filtered.iloc[i - lookback_days:i]
            lookback_returns = np.log(lookback_data / lookback_data.shift(1)).dropna()
            mean_returns     = lookback_returns.mean() * 252
            cov_matrix       = lookback_returns.cov() * 252

            # Check covariance matrix dimensions
            if cov_matrix.shape[0] != len(df.columns) or cov_matrix.shape[1] != len(df.columns):
                raise ValueError("Covariance matrix dimensions are unexpected.")

            # Append covariance matrix to list
            covariances.append(cov_matrix.values)

            # Calculate portfolio values
            cur_portfolio_value    = portfolio_values[-1]
            static_portfolio_value = static_portfolio_values[-1]
            sp500_portfolio_value  = sp500_portfolio_values[-1]

            # Update weights
            optimal_result = max_sharpe(mean_returns, cov_matrix, risk_free, old_weights)
            new_weights    = optimal_result.x

            # Update portfolio values
            portfolio_values.append(cur_portfolio_value * (1 + np.sum(old_weights * lookback_returns.iloc[-1])))
            static_portfolio_values.append(static_portfolio_value * (1 + np.sum(static_weights * lookback_returns.iloc[-1])))
            sp500_portfolio_values.append(sp500_portfolio_value * (1 + sp500_filtered.pct_change().iloc[i]))

            # Update old weights to new weights
            old_weights = new_weights.copy()
        
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            continue
    
    final_dynamic_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    final_static_return  = (static_portfolio_values[-1] / static_portfolio_values[0]) - 1
    final_sp500_return   = (sp500_portfolio_values[-1] / sp500_portfolio_values[0]) - 1

    results = [final_dynamic_return * 100, final_static_return * 100, final_sp500_return * 100, cov_matrix.mean().mean()]
    
    return results

# Main function to run simulations
def main():
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Start timer
    start_time = time.time()
    
    # Download tickers from a CSV file
    tickers_csv = "filtered_tickers.csv"
    tickers = pd.read_csv(tickers_csv)["Ticker"].tolist()
    
    # Download historical data for all tickers for the last 10 years
    start_date = date.today() - timedelta(days=365 * 10)
    end_date   = date.today()
    all_data, _ = get_data(tickers, start=start_date, end=end_date, interval="1d")
    
    # Download S&P 500 data for comparison
    sp500_ticker = ["^GSPC"]
    sp500, _     = get_data(sp500_ticker, start=start_date, end=end_date, interval="1d")
    
    # Run simulations
    simulations = 1000
    results     = []

    for i in range(simulations):
        print("Iteration:", i + 1)
        try:
            # Select random portfolio
            portfolio = random.sample(tickers, random.randint(1, 100))

            # Randomly select start and end dates within the last 10 years
            random_start_day = random.randint(0, (end_date - start_date).days)
            random_end_day   = random.randint(random_start_day, (end_date - start_date).days)

            start_day = start_date + timedelta(days=random_start_day)
            end_day   = start_date + timedelta(days=random_end_day)

            # Determine interval based on the day gap
            day_gap = (end_day - start_day).days
            if day_gap < 7: interval = "1m"
            
            elif 7 <= day_gap < 30: interval = "1h"
            
            else: interval = "1d"

            # Random lookback period (between 3 and 400 days)
            max_lookback = min(abs(random_end_day - random_start_day) - 3, 400)
            lookback = random.randint(3, max_lookback)
            
            # Perform portfolio simulation
            cur_res = portfolio_simulation(all_data[portfolio], sp500, start_day, end_day, lookback, interval)
            cur_res.append(lookback)
            cur_res.append(len(portfolio))
            cur_res.append(random_start_day)
            cur_res.append(random_end_day)
            cur_res.append(interval)
            results.append(cur_res)
            print(cur_res)

            # Save results to CSV periodically
            if i % 100 == 0:
                print("Time elapsed:", round((time.time() - start_time) / 60), "minutes")
                results_df = pd.DataFrame(results, columns=['Dynamic Return (%)', 'Static Return (%)', 'S&P 500 Return (%)', 'Average Cov', 'Lookback Days', 'Portfolio Size', 'Start Date', 'End Date', 'Interval'])
                print(results_df.head())
                results_df.to_csv("Result.csv", index=False)
        
        except Exception as e:
            print(e)
            continue
    
    # Save final results to CSV
    results_df = pd.DataFrame(results, columns=['Dynamic Return (%)', 'Static Return (%)', 'S&P 500 Return (%)', 'Lookback Days', 'Portfolio Size', 'Start Date', 'End Date', 'Interval'])
    print(results_df)
    results_df.to_csv("Result2.csv", index=False)

if __name__ == "__main__":
    main()
