#!/usr/bin/python3

import backtrader.indicators as btind
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm

from dt_help import Helper
from scipy.stats import norm

class MonteCarlo():
    def __init__(
        self,
        data: pd.DataFrame,
        data_true: pd.DataFrame
        ):
        self.data = data
        self.data_true = data_true
        self.intervals = 20
        self.iterations = 1000

    def get_vars(self):
        log_returns = np.log(1 + self.data['close'].pct_change().dropna())
        u = log_returns.mean()    # Mean of the logarithmic return
        var = log_returns.var()   # Variance of the logarithmic return
        drift = u - (0.5 * var)   # drift / trend of the logarithmic return
        stdev = log_returns.std() # Standard deviation of the log return

        daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(self.intervals, self.iterations)))
        self.log_returns = log_returns
        self.daily_returns = daily_returns
        
    def get_preds(self):
        S0 = self.data['close'].iloc[-1]
        price_list = np.zeros_like(self.daily_returns)
        price_list[0] = S0
        
        for t in range(1, self.intervals):
            price_list[t] = price_list[t - 1] * self.daily_returns[t]
        price_list = pd.DataFrame(price_list)
        price_list['close'] = price_list[0]
        self.price_list = price_list

        close = pd.DataFrame(self.data['close'])
        frames = [close,self.price_list]
        self.monte_carlo_forecast = pd.concat(frames)
        self.monte_carlo_forecast.index = close.index.append(pd.bdate_range(start=close.index[-1],periods=self.intervals))
        
    def get_plot(self):
        monte_carlo = self.monte_carlo_forecast.iloc[:,:]
        mask = (monte_carlo.index >= pd.to_datetime(monte_carlo.index[-50]))
        monte_carlo_mask = monte_carlo.loc[mask]

        selected = monte_carlo_mask.columns[monte_carlo_mask.gt(0).any()]
        plt.figure(figsize=(32,16))
        plt.plot(monte_carlo_mask.index,monte_carlo_mask[selected])

        data_true = self.data_true
        mask_true = (data_true.index >= pd.to_datetime(data_true.index[-50]))
        data_true_mask = data_true.loc[mask_true]
        plt.plot(data_true_mask.index,data_true_mask['close'],color='k')
        
        plt.show()        
        
