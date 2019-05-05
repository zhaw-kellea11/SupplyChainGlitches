import quandl
import datetime
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import sys

from sklearn import linear_model
from scipy import stats

QUANDL_API = 'AyLzjcSyz1MzAUVK655-'
quandl.ApiConfig.api_key = QUANDL_API


def quandl_data(stocks, exchange_id, csv=True, suffix=''):
    data = []
    print(stocks)
    if exchange_id == 'FSE':
        stocks = ['FSE/' + s + '_X' for s in stocks if isinstance(s, str)]
    elif exchange_id == 'XFRA':
        stocks = ['XFRA/' + s for s in stocks if isinstance(s, str)]
    else:
        raise Exception('Unknown exchange_id: ' + exchange_id)

    stocks = stocks
    print(stocks)
    print('Querying ' + str(len(stocks)) + ' datasets')
    data = quandl.get(stocks, start_date='2000-01-01', end_date='2019-01-01', collapse='daily', column_index=4)
    print(data)

    if csv:
        data.to_csv('data/stocks/' + exchange_id + suffix + '.csv')
    return data


class StockData:
    def __init__(self):
        self.general = pd.read_csv('data/stocks/XFRA_GENERAL.csv')
        self.prime = pd.read_csv('data/stocks/XFRA_PRIME.csv')

        self.cdax = pd.read_csv('data/stocks/CDAX.csv', sep=';')

        self.cdax = self.cdax[['Date', 'Close']]

        self.general.columns = [c.split('/', 1)[-1].split(' -', 1)[0] for c in self.general.columns]
        self.prime.columns = [c.split('/', 1)[-1].split(' -', 1)[0] for c in self.prime.columns]
        self.general_tickers = list(self.general.columns)[1:]
        self.prime_tickers = list(self.prime.columns)[1:]

        self.prime['Date'] = pd.to_datetime(self.prime['Date'], format='%Y-%m-%d')
        self.general['Date'] = pd.to_datetime(self.general['Date'], format='%Y-%m-%d')
        self.cdax['Date'] = pd.to_datetime(self.cdax['Date'], format='%Y-%m-%d')

        self.bdays = list(self.cdax['Date'])

        self.prime = self.prime[self.prime['Date'].isin(self.bdays)]
        self.general = self.general[self.general['Date'].isin(self.bdays)]

        self.prime = self.prime.set_index(['Date']).pct_change().iloc[1:].reset_index(level=0)
        self.general = self.general.set_index(['Date']).pct_change().iloc[1:].reset_index(level=0)
        self.cdax = self.cdax.set_index(['Date']).pct_change().iloc[1:].reset_index(level=0)

    def ticker2data(self, ticker):
        if ticker in self.general_tickers:
            stock_data = self.general[['Date', ticker]]
        elif ticker in self.prime_tickers:
            stock_data = self.prime[['Date', ticker]]
        else:
            raise Exception('Unmappable Ticker: ' + str(ticker))
        return stock_data

    def extractWindows(self, filter, window=40):
        windows = []
        glitches = []
        for index, glitch in filter.iterrows():
            try:
                stock_data = self.ticker2data(glitch['Trading'])
            except:
                continue

            origin = glitch['Datum']
            while not origin in self.bdays:
                origin = origin + datetime.timedelta(days=1)

            window_days = [origin]
            next_day = origin
            for i in range(window):
                while (next_day in window_days) or (next_day not in self.bdays):
                    next_day = next_day + datetime.timedelta(days=1)
                window_days.append(next_day)

            previous_day = origin
            for i in range(window):
                while (previous_day in window_days) or (previous_day not in self.bdays):
                    previous_day = previous_day - datetime.timedelta(days=1)
                window_days.append(previous_day)

            window_data = stock_data[stock_data['Date'].isin(window_days)]
            isna = window_data.isnull().values.any()
            if not isna:
                windows.append(window_data)
                glitches.append(glitch)
        return windows, glitches

    def abnormal_return(self, windows):
        abnormal_returns = []
        for window in windows[:]:
            ticker = list(window.columns)[-1]
            event_dates = list(window['Date'])
            event_date = event_dates[len(event_dates) // 2]
            stock_data = self.ticker2data(ticker)

            max_date = event_date - datetime.timedelta(days=10)

            # max_date = max(window['Date']) + datetime.timedelta(days=100)
            #independent_stock_history = stock_data[(~stock_data['Date'].isin(
            #    event_dates)) & (stock_data['Date'] <= max_date)].dropna(how='any')
            independent_stock_history = stock_data[(stock_data['Date'] <= max_date)].dropna(how='any')

            min_date = min(independent_stock_history['Date'])

            event_cdax = self.cdax[self.cdax['Date'].isin(event_dates)]

            # independent_cdax_history = self.cdax[~self.cdax['Date'].isin(event_dates)]
            independent_cdax_history = self.cdax[(
                self.cdax['Date'] >= min_date) & (self.cdax['Date'] <= max_date)]

            #self.stock_array = np.array(independent_stock_history.iloc[:, 1])[-200:]
            #self.cdax_array = np.array(independent_cdax_history.iloc[:, 1])[-200:]

            self.stock_array = np.array(independent_stock_history.iloc[:, 1])[:]
            self.cdax_array = np.array(independent_cdax_history.iloc[:, 1])[:]

            self.design_matrix = np.stack((np.ones(len(self.stock_array)), self.cdax_array), axis=1)

            clf = linear_model.LinearRegression(fit_intercept=False)
            clf.fit(self.design_matrix, self.stock_array)

            # Generate fitted data
            fitted = self.fitFunction(*clf.coef_)

            # Calculate Residuals (delta-dof is 2, one for each fitted param)
            residuals = fitted - self.stock_array
            residuals_std = residuals.std(ddof=2)

            variances = residuals_std**2

            # Calculate variances
            variances = residuals_std**2 * np.linalg.inv(np.dot(self.design_matrix.T, self.design_matrix))
            std_coef = np.sqrt(np.diagonal(variances))

            # Calculate t- and p-values
            t_coef = clf.coef_ / std_coef
            p_coef = [2 * (1 - stats.t.cdf(np.abs(coef), len(self.design_matrix) - 3)) for coef in t_coef]

            model = np.stack((clf.coef_, std_coef, t_coef, p_coef))

            # self.printModel(model)

            beta = clf.coef_[1]
            intercept = clf.coef_[0]

            stock_event_price = np.array(window.iloc[:, 1])
            cdax_event_price = np.array(event_cdax.iloc[:, 1])

            expected_return = intercept + beta * cdax_event_price
            abnormal_return = stock_event_price - expected_return

            #abnormal_return_rel = stock_event_price / expected_return - 1
            #abnormal_return_rel = np.array(pd.DataFrame(abnormal_return).pct_change())[1:]

            # print(abnormal_return_rel)
            # self.printModel(model)
            package = [abnormal_return, ticker, event_dates, model]
            abnormal_returns.append(package)
        return abnormal_returns

    def fitFunction(self, a, b1):
        return a + b1 * self.cdax_array

    @staticmethod
    def printModel(model):
        """ Print model parameters and statistics

        statistics and parameters are printed to console

        Args:
            model (np.array(float)): model data of the regression

        """
        print()
        print('--- Coefficients ---')
        print('Beta:', round(model[0][1], 3))
        print('Intercept:', round(model[0][0], 3))
        print('------- STD --------')
        print('Beta:', round(model[1][1], 3))
        print('Intercept:', round(model[1][0], 3))
        print('----- t-value ------')
        print('Beta:', round(model[2][1], 3))
        print('Intercept:', round(model[2][0], 3))
        print('----- p-value ------')
        print('Beta:', model[3][1])
        print('Intercept:', model[3][0])
