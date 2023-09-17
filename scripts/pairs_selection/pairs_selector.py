from scripts.utils.util_classes import CustomLogger
from scripts.utils.util_functions import normalise_prices
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
from sklearn.manifold import MDS

logger = CustomLogger.get_logger(name=__name__)


# parent class to choose pairs for pairs trading
class pairsSelector:
    def __init__(self, df, time_period, top_n, options={}):
        self.df = df[-time_period:]
        self.top_n = top_n
        self.pairs = []
        self.options = options
        self.normalised_df = df

    def get_pairs(self):
        raise NotImplementedError

    def plot_similarity_metrics(self):
        raise NotImplementedError

    def plot_normalised_pair(self, pair):
        pair_df = self.normalised_df[[pair[0], pair[1]]]
        return px.line(pair_df.melt(ignore_index=False).reset_index(), x='date', y='value', color='ticker',
                       title='Normalised prices for pair {} with {} normalisation'.format(pair, self.options[
                           'price_normalisation']))


class distanceSelector(pairsSelector):
    def __init__(self, df, time_period, top_n,
                 options={'distance_metric': 'SSD', 'price_normalisation': 'min_max'}):
        super().__init__(df, time_period, top_n, options=options)

    def get_pairs(self):
        metric = self.options['distance_metric']
        self.normalised_df = normalise_prices(self.df, method=self.options['price_normalisation'])

        if metric == 'SSD':
            # Calculate sum of squared distances between each pair of tickers
            n = len(self.normalised_df.columns)

            distance_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(i + 1, n):
                    distance_matrix[i, j] = np.sum((self.normalised_df.iloc[:, i] - self.normalised_df.iloc[:, j]) ** 2)
                    distance_matrix[j, i] = distance_matrix[i, j]

            distance_df = pd.DataFrame(distance_matrix, index=self.normalised_df.columns,
                                       columns=self.normalised_df.columns)

        else:
            raise NotImplementedError

        melted_df = distance_df.reset_index().melt(id_vars='ticker', var_name='ticker2', value_name='distance')
        melted_df = melted_df.rename(columns={'ticker': 'ticker1'})

        # Remove zero distances (self-pairs) and duplicates
        melted_df = melted_df[melted_df['distance'] > 0]
        melted_df['ticker_pair'] = melted_df.apply(lambda row: list(sorted([row['ticker1'], row['ticker2']])), axis=1)
        melted_df = melted_df.drop_duplicates(subset=['ticker_pair'])

        # Sort by distance and select the closest pairs
        melted_df = melted_df.sort_values(by='distance')
        closest_pairs = melted_df.head(self.top_n)['ticker_pair'].values.tolist()
        self.distance_df = distance_df

        return closest_pairs

    def plot_summary(self, pairs):
        tickers = list(set([ticker for pair in pairs for ticker in pair]))
        # Plot plotly heatmap from distance_matrix with tickers as labels with meaningful colormap for distance
        distance_df = self.distance_df.loc[tickers, tickers]

        combined_fig = make_subplots(rows=1, cols=2 , subplot_titles=
        ['Distance matrix for {} tickers with {} metric'.format(len(tickers), self.options['distance_metric']),
         "Multidimensional Scaling of tickers based on Distance Matrix"
         ])

        fig1 = px.imshow(distance_df, color_continuous_scale='RdBu_r')

        # Apply MDS
        mds = MDS(n_components=2, dissimilarity='precomputed' ,normalized_stress='auto')
        data_2d = mds.fit_transform(distance_df)

        # Create a DataFrame for 2D data for easy plotting
        data_2d_df = pd.DataFrame(data_2d, columns=['x', 'y'])
        data_2d_df['ticker'] = distance_df.index

        # Create the Plotly figure
        fig2 = px.scatter(data_2d_df, x='x', y='y', text='ticker')
        fig2.update_layout(

            xaxis_title="MDS 1",
            yaxis_title="MDS 2",
        )

        for trace in fig1['data']:
            combined_fig.add_trace(trace, row=1, col=1)
        for trace in fig2['data']:
            combined_fig.add_trace(trace, row=1, col=2)

        combined_fig.update_layout(template = 'plotly_dark')
        return combined_fig


class cointergrationSelector(pairsSelector):
    def __init__(self, df, time_period, top_n,
                 options={'pval_threshold': 0.05, 'price_normalisation': None}):
        super().__init__(df, time_period, top_n, options=options)

    def get_pairs(self):
        self.normalised_df = normalise_prices(self.df, method=self.options['price_normalisation'])
        # Find cointegrated pairs using Engle-Granger test
        # First this performs an OLS regression on the two series
        # Then it performs an ADF test on the residuals
        # If residual is stationary then the two series are cointegrated

        # Calculate cointegration test for all pairs
        n = len(self.normalised_df.columns)
        pvalue_matrix = np.ones((n, n))
        tickers = self.normalised_df.columns
        pairs = []
        pairs_spread_half_life = []
        for i in range(n):
            for j in range(n):
                logger.info('Calculating pair {} out of {} pairs'.format(i * n + j, n ** 2))
                logger.info('Calculating cointegration test for {} and {}'.format(tickers[i], tickers[j]))
                p1 = self.normalised_df.iloc[:, i]
                p2 = self.normalised_df.iloc[:, j]

                coint_t, p_val, c_val = sm.tsa.stattools.coint(p1, p2)

                pvalue_matrix[i, j] = p_val

                if p_val < self.options['pval_threshold'] and i != j:
                    pairs.append((tickers[i], tickers[j], p_val))

                    # Calculate half life of spread

                    # Implement OLS regression
                    x = sm.add_constant(self.normalised_df.iloc[:, i]).values
                    y = self.normalised_df.iloc[:, j].values
                    model = sm.OLS(y, sm.add_constant(x))

                    results = model.fit()
                    residuals = results.resid

                    # Construct lagged residuals series
                    lagged_residuals = residuals[:-1]
                    residuals = residuals[1:]

                    # Run OLS regression on residuals and lagged residuals
                    model = sm.OLS(residuals, sm.add_constant(lagged_residuals))
                    results = model.fit()
                    phi = results.params[1]
                    half_life = -np.log(2) / phi
                    pairs_spread_half_life.append((tickers[i] , tickers[j] , half_life))

        pvalue_df = pd.DataFrame(pvalue_matrix, index=self.normalised_df.columns,
                                 columns=self.normalised_df.columns)
        self.pvalue_df = pvalue_df

        self.half_life_df = pd.DataFrame(pairs_spread_half_life , columns = ['ticker1' , 'ticker2' , 'half_life'])


        # Choose top_n pairs with lowest p-values
        closest_pairs = sorted(pairs, key=lambda x: x[2])[:self.top_n]
        closest_pairs = [[pair[0], pair[1]] for pair in closest_pairs]

        return closest_pairs

    def plot_summary(self, pairs):
        tickers = list(set([ticker for pair in pairs for ticker in pair]))
        # Plot plotly heatmap from distance_matrix with tickers as labels with meaningful colormap for distance
        pvalue_df = self.pvalue_df.loc[tickers, tickers]

        combined_fig = make_subplots(rows=1, cols=2 , subplot_titles=
        ['p-value matrix for {} tickers with Engle-Granger test'.format(len(tickers)),
         'p-value vs half life for selected pairs'
         ])

        fig1 = px.imshow(pvalue_df, color_continuous_scale='RdBu_r')


        for trace in fig1['data']:
            combined_fig.add_trace(trace, row=1, col=1)

        pval_df = self.pvalue_df.melt(ignore_index=False , var_name = 'ticker2' , value_name = 'pval')
        pval_df = pval_df.reset_index().rename(columns = {'ticker' : 'ticker1'})

        df = self.half_life_df.merge(pval_df , on = ['ticker1' , 'ticker2'])
        df['ticker_pair'] = df.apply(lambda row: list(sorted([row['ticker1'], row['ticker2']])), axis=1)

        fig2 = px.scatter(df , x = 'pval' , y = 'half_life' , text = 'ticker_pair')


        # Add text annotations
        for index, row in df.iterrows():
            fig2.add_annotation(
                x=row['pval'],
                y=row['half_life'],
                text=str(row['ticker_pair']),
                showarrow=False,
                font=dict(size=10)
            )

        for trace in fig2['data']:
            combined_fig.add_trace(trace, row=1, col=2)


        combined_fig.update_layout(template = 'plotly_dark')
        combined_fig.update_xaxes(title_text="p-value" , row = 1 , col = 2)
        combined_fig.update_yaxes(title_text="Half life" , row = 1 , col = 2)
        return combined_fig


class correlationSelector(pairsSelector):
    def __init__(self, df, time_period, top_n,
                 options={'pval_threshold': 0.05, 'price_normalisation': None}):
        super().__init__(df, time_period, top_n, options=options)

    def get_pairs(self):
        self.normalised_df = normalise_prices(self.df, method=self.options['price_normalisation'])
        # Get correlation matrix
        corr = self.normalised_df.corr()

        melted_df = corr.reset_index().melt(id_vars='ticker', var_name='ticker2', value_name='correlation')
        melted_df = melted_df.rename(columns={'ticker': 'ticker1'})

        # Remove correlation 1 values (self-pairs)
        melted_df = melted_df[melted_df['correlation'] < 1]
        melted_df['ticker_pair'] = melted_df.apply(lambda row: list(sorted([row['ticker1'], row['ticker2']])), axis=1)
        melted_df = melted_df.drop_duplicates(subset=['ticker_pair'])

        # Sort by distance and select the closest pairs
        melted_df = melted_df.sort_values(by='correlation')
        closest_pairs = melted_df.head(self.top_n)['ticker_pair'].values.tolist()
        self.corr_df = corr

        return closest_pairs


if __name__ == '__main__':
    from scripts.db.db_interface import dbInterface
    from scripts.utils.util_functions import read_config

    config = read_config()
    tickers_collection_name = config['database']['collections']['tickers_collection_name']
    price_collection_name = config['database']['collections']['price_collection_name']
    batch_size = config['database']['batch_size']
    start_data = config['data']['start_data']
    end_data = config['data']['end_data']

    db = dbInterface(config=config['database'], ticker_collection=tickers_collection_name,
                     price_collection=price_collection_name)

    prices = db.retrieve_prices(['MSFT', 'AAPL', 'AMZN', 'NVDA', 'META', 'AMD', 'KO', 'PEP', 'XOM', 'CVX'])[
        ['adj close']].reset_index()
    prices = prices.pivot_table(index='date', columns='ticker', values='adj close')

    selector = correlationSelector(prices, time_period=400, top_n=10)

    closest_pairs = selector.get_pairs()
    closest_pair = closest_pairs[0]
    selector.plot_normalised_pair(pair=closest_pair)
    print('Done')

    # selector = distanceSelector(prices , time_period=100 , top_n = 10)
    #
    # closest_pairs = selector.get_pairs()
    # closest_pair = closest_pairs[0]
    # fig = selector.plot_normalised_pair(pair = closest_pair)
    #
    # # Convert list of ticker pairs to list of unique tickers
    # closest_tickers = list(set([ticker for pair in closest_pairs for ticker in pair]))
    # fig = selector.plot_similarity_metrics( closest_tickers)

    print('done')
