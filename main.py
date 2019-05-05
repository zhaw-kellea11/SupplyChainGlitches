from glitchData import Companies
from stockData import quandl_data, StockData
from analytics import abnormal_analytics


def main():
    print('Initializing')
    fn_glitches = 'data/Glitches_all.xlsx'
    companies = Companies(fn_glitches=fn_glitches)
    stocks = StockData()

    n_rand = 10
    # companies_random = [Companies(randomized=companies) for _ in range(n_rand)]

    print('Building Event windows...')
    windows, glitches = stocks.extractWindows(companies.glitches)
    print('Calculating abnormal returns...')
    abnormal_returns = stocks.abnormal_return(windows)

    # abnormal_analytics(abnormal_returns)

    categories = ['Sector', 'Ursache', 'Grund (Art der Störung)', 'Verantwortlich (Quelle)']
    for cat in categories:
        values = set([g[cat] for g in glitches])
        values = {v: [] for v in values}
        print(values)
        for i, g in enumerate(glitches):
            values[g[cat]].append(abnormal_returns[i])

        for v, r in values.items():
            fig_name = ('returns_' + cat + '_' + v + '.pdf').replace('/', '-')
            fig_title = 'Abnormal returns around glitch event (' + cat + ': ' + v + ')'
            abnormal_analytics(r, fig_name, fig_title)




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
