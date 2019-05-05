from glitchData import Companies
from stockData import quandl_data, StockData
from analytics import abnormal_analytics


def main():
    print('Initializing')
    # fn_glitches = 'data/Glitches_all_ohne_streik.xlsx'
    fn_glitches = 'data/Glitches_all.xlsx'
    companies = Companies(fn_glitches=fn_glitches)
    stocks = StockData()

    print('Building Event windows...')
    windows, glitches = stocks.extractWindows(companies.glitches)
    print('Calculating abnormal returns...')
    abnormal_returns = stocks.abnormal_return(windows)

    abnormal_analytics(abnormal_returns)

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


if __name__ == '__main__':
    main()
