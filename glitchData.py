import pandas as pd
import random as rand
rand.seed()


class GlitchData:
    def __init__(self, fn='data/data.xlsx'):
        self.data = pd.read_excel(fn, sheet_name='Störungen', header=2)
        self.tickers = list(self.data['Trading'].unique())

        self.glitches = {}
        for tick in self.tickers:
            self.glitches[tick] = self.data[self.data['Trading'] == tick]['Datum'].to_list()

        print(self.glitches)


class Companies:
    def __init__(self, fn='data/data.xlsx', randomized=False):
        if not randomized:
            excluded_categories = ['Banks',
                                   'Financial Services',
                                   'Insurance',
                                   'Media',
                                   'Software',
                                   'Telecommunication']

            self.prime = pd.read_excel(fn, sheet_name='Prime Standard', header=7)
            self.prime = self.prime[~self.prime['Sector'].isin(excluded_categories)]
            self.general = pd.read_excel(fn, sheet_name='General Standard', header=7)
            self.general = self.general[~self.general['Sector'].isin(excluded_categories)]

            self.tickers_prime = list(self.prime['Trading Symbol'].unique())
            self.tickers_general = list(self.general['Trading Symbol'].unique())
            self.tickers_general = [t for t in self.tickers_general if t != 'n.a']

            self.glitches = pd.read_excel(fn, sheet_name='Störungen', header=2)
            self.glitches = self.glitches.dropna(subset=['Relevant?'])

            self.glitches = self.glitches[(self.glitches['Datum'] >= '2007-03-01')
                                          & (self.glitches['Datum'] <= '2018-10-01')]

        else:
            self.glitches_rand = pd.DataFrame(columns=['Datum', 'Trading'])
            glitches_rand_list = []
            for _ in range(len(randomized.glitches)):
                trading = rand.choice(randomized.tickers_general + randomized.tickers_prime)
                date = self.random_dates(pd.to_datetime('2007-03-01'), pd.to_datetime('2018-10-01'))
                glitches_rand_list.append([date, trading])

            self.glitches_rand = pd.DataFrame(glitches_rand_list, columns=['Datum', 'Trading'])
            self.glitches = self.glitches_rand
        # self.tickers = self.tickers_prime + self.tickers_general

    def random_dates(self, start, end, n=1):
        start_u = start.value // 10**9
        end_u = end.value // 10**9
        return pd.to_datetime(rand.randint(start_u, end_u), unit='s').replace(hour=0, minute=0, second=0)
