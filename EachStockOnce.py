# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 14:28:20 2025

@author: Arapi
"""

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
stocks = ['AAPL' , 'NVDA', 'MSFT' , 'AMZN', 'META', 'GOOGL', 'GOOG', 'AVGO', 'TSLA', 'TPE:2330', 'LLY', 'JPM',
          'BRK.B', 'V', 'MA', 'COST', 'WMT', 'UNH', 'NFLX', 'PG', 'HKG:0700', 'JNJ', 'HD', 'ABBV', 'BAC', 'ETR.SAP', 
          'KO', 'CPH:NOVO.B', 'AMS:ASML', 'CRM', 'ORACL', 'HKG:9988', 'CVX', 'WFC', 'SWX:NESN', 'CSCO', 'PM', 'ABT', 
          'SWX:ROG', 'MRK', 'IBM', 'LON:AZN', 'GE', 'LIN', 'LON:HSBA', 'SWX:NOVN', 'MCD', 'ACN', 'PEP', 'LON:SHEL',
          'ISRG', 'DIS', 'TMO', 'EPA:MC', 'T', 'ADBE', 'GS', 'NOW', 'VZ', 'TYO:7203', 'ETR:SIE', 'RTX', 'PLTR','TXN',
          'TXN', 'KRX:005930', 'QCOM', 'AXP', 'INTU', 'SPGI', 'AMGN', 'BKNG', 'TSX:RY', 'PGR', 'ASX:CBA', 'CAT', 'AMD',
          'MS', 'BSX', 'TYO:6758', 'PFE', 'UNP', 'C', 'NEE', 'GILD', 'TMUS', 'BLK', 'TYO:8306', 'LON:ULVR',
          'UBER', 'LOW', 'ETR:ALV', 'TJX', 'FI', 'CMCSA', 'HON', 'SYK', 'XTSLA', 'EPA:TTE', 'DHR', 'SCHW', 'ETR:DTE',
          'SBUX', 'SHOP.NE', 'ADP', 'EPA:SU', 'BA', 'EPA:SAN', 'AMAT', 'VRTX', 'ASX:BHP', 'COP', 'BMY', 'PANW',
          'MDT', 'DE', 'SWX:UBSG', 'TYO:6501', 'ADI', 'PLD', 'MMC', 'BX', 'ETN', 'SWX:CFR', 'CB', 'NSE:HDFCBANK',
          'HKG:1810', 'EPA:AI', 'EPA:AIR', 'BME:SAN', 'TSX:TD', 'HKG:3690', 'MU', 'MO', 'ICE', 'LRCX', 'AMT', 'INTC',
          'SO', 'WELL', 'SWX:ZURN', 'LMT', 'TYO:8316', 'MELI', 'CRWD', 'LON:BP', 'WM', 'LON:REL', 'TSX:ENB', 'KLAC',
          'NKE','ANET', 'SPOT', 'BME:IBE', 'ELV', 'CME', 'EPA:SAF', 'TYO:6098', 'DUK', 'EPA:RMS', 'EPA:OR', 'EQIX',
          'GEV','MDLZ', 'CI', 'EPA:EL', 'SWX:ABBN', 'UPS', 'SHW', 'AJG', 'HKG:1299', 'MMM', 'PH','MCK', 'LON:RR',
          'BIT:UCG', 'KKR', 'MCO', 'ORLY','LON:BATS', 'CVS', 'NSE:RELIANCE', 'BME:BBVA', 'ASX:CSL', 'EPA:BNP', 'AON',
          'TYO:7974', 'ETR:MUV2', 'TT', 'TYO:6861', 'PDD','HKG:0939', 'BN.NE', 'ZTS', 'TDG', 'CTAS', 'APH', 'V', 'ITW', 
          'NSE:ICICIBANK', 'LON:GSK', 'PNC', 'MRVL', 'TSX:BMO', 'CMG', 'COF', 'REGN', 'CEG', 'LON:RIO','EPA:CS', 'USB',
          'CL', 'MSI', 'LON:LSEG', 'KRX:000660', 'APP', 'WMB', 'TSX:CSU', 'CRH', 'TSX:CP', 'FTNT', 'ECL', 'TYO:8766',
          'DASH', 'APD', 'HLT', 'ASX:NAB','MAR', 'TYO:8035', 'TPE:2454','BDX', 'CDNS', 'EMR', 'NOC', 'TYO: 8411'
          'SGX:D05', 'TPE:2317', 'BK', 'NSE:INFY', 'ROP', 'LON:BA', 'AMS: PRX', 'AMS: INGA', 'SPG', 'ABNB', 'TFC', 'LON:LLOY', 
          'TSX:BNS', 'HCA', 'CSX', 'EPA:DG', 'LON:BARC', 'AZO', 'RCL', 'LON:BARC', 'AZO', 'RCL', 'FDX', 'TYO:9983', 
          'LON: GE', 'BIT:RACE', 'AFL', 'LON:CPG', 'TRV', 'TSX:CNQ', 'ADSK', 'GD', 'APO', 'AEP', 'MSTR', 'LON:NG'
          'KMI', 'TSX:MFC', 'TSX:CM', 'JCI', 'TYO:8058', 'SLB', 'OKE', 'BIT:ENEL', 'ASX:ANZ', 'BME: ITX', 'NXPI',
          'TGT',  'NSC', 'WDAY', 'PCAR', 'HKG:0388', 'STO:INVE.B', 'TYO:4063', 'STO:VOLV.B', 'SWX:HOLN', 'TADAWUL:1120',
          'AMP', 'TYO:9984', 'ASX:WES', 'LON: III', 'GM', 'EBR: ABI', 'AIG', 'TYO: 8001', 'TSX: CNR', 'ALL', 'PSX',
          'RSG',  'ETR: RHM', 'DLR', 'MET', 'SNOW', 'O', 'CARR', 'HWM', 'ASX: MQG', 'TYO: 9433', 'WCN', 'FCX', 'ETR: DB1',
          'LNG', 'HKG: 1398', 'MPC', 'ETR: IFX', 'CMI', 'PAYX', 'SWX: SREN', 'PSA', 'HKG: 1211', 'NEM', 'FLUT', 'TYO: 8031',
          'DFS', 'STO: ATCO.A', 'ETR: BAS', 'SWX: LONN', 'D', 'ETR: MBG', 'KMB', 'SRE', 'HKG: 9618', 'TYO: 4502', 
          'CPRT', 'TSX: SU',  'TSX: AEM', 'MSCI', 'NSE: BHARTIARTL', 'KVUE', 'SWX: ALC', 'TEL', 'FICO', 'EPA: SGO',
          'TSX: TRP', 'EPA: BN', 'SE', 'ROST', 'LON: RKT', 'GWW', 'JSE: NPN', 'COR', 'SGX: O39', 
          'LON: HLN', 'LON: EXPN', 'EXC', 'TYO: 7011', 'TEAM', 'HEL: NDA.FI', 'CBRE', 'TYO: 4568', 'NET',
          'BKR', 'VRSK', 'CTSH', 'FAST', 'VST', 'YUM', 'KR', 'SWX: GIVN', 'JPY', 'HKG: 3988', 'TADAWUL: 2222', 'AME',
          'EW', 'ETR: DBK', 'TYO: 7741', 'ETR: ADS', 'ETR: DHL', 'PRU', 'CPH: DSV', 'TYO: 6857', 'TYO: 7267',
          'URI', 'LON: GLEN', 'CCI', 'VLO', 'CTVA', 'AMS: ADYEN', 'XEL', 'COIN', 'AMS: WKL', 'MNST', 'OTIS',
          'FIS', 'PEG', 'TRGP', 'HES', 'KDP', 'TYO: 9434', 'SWX: SIKA', 'LHX', 'GEHC', 'GLW', 'TSX: ATD', 'AXON', 'LULU', 
          'EBR: ARGX', 'HKG: 9999', 'LON: STAN', 'LON: NWG', 
          'HIG', 'IT', 'ETR: ENR', 'A', 'SYY', 'LON: AAL', 'DHI', 'ASX: GMG', 'ETR', 'TSX: IFC', 'TTWO',
          'HKG: 2318', 'IDXX', 'PWR', 'F', 'ED', 'NSE: TCS', 'WTW', 'NDAQ', 'GRMN', 'FERG', 'SGX: U11',
          'ACGL', 'CHTR', 'SWX: PGHN', 'XYZ', 'GIS', 'WEC', 'DXCM', 'IR', 'TYO: 6702', 'BME: AMS', 'RMD', 'AMS: AD', 
          'HUBS', 'IQV', 'TYO: 4519', 'LON: TSCO', 'TYO: 7751', 'BIT: G', 'AVB', 'VEEV', 'DD',  'NU', 'EXR', 
          'PCG', 'LON: IMB', 'ODFL', 'VICI', 'GBP', 'BIT: ENI', 'EBAY', 'EA', 'TSX: NA', 'ARES', 'TSX: SLF', 
          'WAB', 'AUD', 'CAH', 'MTB', 'ALNY', 'DDOG', 'STO: ASSA.B', 'XYL', 'NUE', 'EPA: ENGI', 'HUM', 'ROK', 'BVMF: VALE3',
          'LE3', 'HKG: 9961', 'TYO: 3382', 'TSX: WPM', 'MCHP', 'TSX: ABX', 'RJF', 'VMC', 'BR', 'VRT', 'CSGP', 'TYO: 6503', 
          'AEE', 'RBLX', 'EFX', 'TYO: 2914']


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
    max_drawdown = calculate_max_drawdown(returns)
    return portfolio_return, portfolio_volatility, sharpe_ratio, max_drawdown, weights

# Value at Risk
def calculate_var(returns, confidence=0.95):
    mean = np.mean(returns)
    std_dev = np.std(returns)
    var = norm.ppf(1 - confidence, mean, std_dev)
    return var

# Maximum Drawdown
def calculate_max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    return max_drawdown

# Hauptschleife
portfolio_results = []

used_stocks = set()
for i in range(num_portfolios):
    available_stocks = list(set(stocks) - used_stocks)
    selected_stocks = generate_portfolio(available_stocks, num_stocks_per_portfolio)
    used_stocks.update(selected_stocks)
    data = yf.download(selected_stocks, start=start_date, end=end_date, auto_adjust=False, actions=False)['Adj Close']
    returns = data.pct_change().drop 