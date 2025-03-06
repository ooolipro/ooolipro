# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:58:11 2025

@author: Arapi
"""

import yfinance as yf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.stats import norm

# Aktienliste (als Beispiel)
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'JPM', 'V']

# Parameter
num_portfolios = 30
num_stocks_per_portfolio = 4
start_date = '2008-03-28'
end_date = '2025-01-31'

# ETF Download
acwi = yf.download('ACWI', start=start_date, end=end_date, auto_adjust=False, actions=False)['Adj Close']
acwi_returns = acwi.pct_change().dropna()

# Portfolios generieren
def generate_portfolio(stocks, num_stocks):
    return random.sample(stocks, num_stocks)


# Portfolio Performance
def calculate_metrics(data):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    weights = np.random.random(len(data.columns))
    weights /= np.sum(weights)
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio, weights

# Value at Risk
def calculate_var(returns, confidence=0.95):
    mean = np.mean(returns)
    std_dev = np.std(returns)
    var = norm.ppf(1 - confidence, mean, std_dev)
    return var

# Hauptschleife
portfolio_results = []

for i in range(num_portfolios):
    selected_stocks = generate_portfolio(stocks, num_stocks_per_portfolio)
    data = yf.download(selected_stocks, start=start_date, end=end_date, auto_adjust=False, actions=False)['Adj Close']
    returns = data.pct_change().dropna()
    port_return, port_volatility, sharpe, weights = calculate_metrics(data)
    var = calculate_var(returns.mean().mean(), confidence=0.95)
    portfolio_results.append((selected_stocks, port_return, port_volatility, sharpe, var))

# Ergebnisse ausgeben
for i, result in enumerate(portfolio_results):
    print(f"Portfolio {i+1}: {result[0]}")
    print(f"Return: {result[1]:.4f}, Volatility: {result[2]:.4f}, Sharpe Ratio: {result[3]:.4f}, VaR: {result[4]:.4f}")
    print("-")

# ETF Vergleich
acwi_var = calculate_var(acwi_returns.mean(), confidence=0.95)
print(f"ETF ACWI: Return: {acwi_returns.mean().mean() * 252:.4f}, VaR: {acwi_var:.4f}")


##VaR: NaN warum?, ETF nur return, ERGEBNIS NICHT REPRODUZIERBAR! IMMER ANDERE ERGEBNISSE



