from glitchData import GlitchData, Companies
from stockData import quandl_data, StockData
from analytics import abnormal_analytics, abnormal_analytics_pct


def main():
    print('Initializing')
    companies = Companies()
    stocks = StockData()

    n_rand = 10
    # companies_random = [Companies(randomized=companies) for _ in range(n_rand)]

    print('Building Event windows...')
    windows = stocks.extractWindows(companies.glitches)
    print('Calculating abnormal returns...')
    abnormal_returns = stocks.abnormal_return(windows)

    abnormal_analytics(abnormal_returns)



    """
    print('Randomizing...')
    randomized_abnomal_returns = []
    for i, c in enumerate(companies_random):
        print(str(i + 1) + '/' + str(n_rand))
        windows = stocks.extractWindows(c.glitches)
        randomized_abnomal_returns.append(stocks.abnormal_return_pct(windows))

    print('Calculating statistics...')
    abnormal_analytics_pct(abnormal_returns, randomized_abnomal_returns)
    """
    # hist_FSE_prime = quandl_data(companies.tickers_prime, exchange_id='FSE', suffix='_PRIME')
    # hist_XFRA_prime = quandl_data(companies.tickers_prime, exchange_id='XFRA', suffix='_PRIME')
    # hist_FSE_general = quandl_data(companies.tickers_general, exchange_id='FSE', suffix='_GENERAL')
    # hist_XFRA_general = quandl_data(companies.tickers_general, exchange_id='XFRA', suffix='_GENERAL')


if __name__ == '__main__':
    main()
