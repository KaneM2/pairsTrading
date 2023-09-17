import statsmodels.api as sm
import pandas as pd
import plotly.express as px
import datetime as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class baseStrategy:
    def __init__(self, price_df, pairs, start_date, end_date, parameters={}):
        self.price_df = price_df.loc[start_date:end_date]
        self.pairs = pairs
        self.parameters = parameters

    # Method to backtest every pair in the portfolio
    def backtest_portfolio(self ,backtest_data , initial_capital = 1000):
        portfolio = pd.DataFrame(index=backtest_data.index)
        portfolio['equity_curve'] = initial_capital

        for pair in self.pairs:
            pair_portfolio = self.backtest_pair(pair = pair , backtest_data = backtest_data)
            portfolio[tuple(pair)] = pair_portfolio['equity_curve']
            portfolio['equity_curve'] += pair_portfolio['equity_curve']


        portfolio['returns'] = portfolio['equity_curve'].pct_change().fillna(0)
        portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod()
        portfolio['drawdown_percentage'] = 100*(portfolio['equity_curve'] - portfolio['equity_curve'].cummax())/portfolio['equity_curve'].cummax()
        self.portfolio = portfolio

        max_drawdown = portfolio['drawdown_percentage'].min()
        sharpe_ratio = portfolio['returns'].mean() / portfolio['returns'].std()
        sortino_ratio = portfolio['returns'].mean() / portfolio[portfolio['returns'] < 0]['returns'].std()


    # Method to backtest a single pair
    def backtest_pair(self):
        raise NotImplementedError

    def plot_portfolio_statistics(self):

        combined_fig = make_subplots(rows=2, cols=2 , subplot_titles=
        ['Equity Curve for strategy',
         'Distribution of Returns for strategy',
         'Drawdown Percentage for strategy',
         'Cumulative Returns for strategy'
         ])

        fig1 = px.line(self.portfolio , x = self.portfolio.index , y = 'equity_curve' , title = 'Equity curve for portfolio')
        fig1.update_layout(template = 'plotly_dark')

        #Cut first 20 values as they are likely to be distorted due to the starting point of 0
        if len(self.portfolio) > 30:
            returns = self.portfolio[['returns']].iloc[30:]
        else:
            returns = self.portfolio[['returns']]

        # Filter out values smaller than -1 and larger than 1 or equal to zero
        returns = returns[(returns['returns'] > -1) & (returns['returns'] < 1) & (returns['returns'] != 0)]

        fig2 = px.histogram(returns, x = 'returns' )

        fig3 = px.line(self.portfolio , x = self.portfolio.index , y = 'drawdown_percentage'
                       , title = 'Drawdown percentage for portfolio')

        fig4 = px.line(self.portfolio , x = self.portfolio.index , y = 'cumulative_returns')


        for trace in fig1['data']:
            combined_fig.add_trace(trace, row=1, col=1)
        for trace in fig2['data']:
            combined_fig.add_trace(trace, row=1, col=2)
        for trace in fig3['data']:
            combined_fig.add_trace(trace, row=2, col=1)
        for trace in fig4['data']:
            combined_fig.add_trace(trace, row=2, col=2)

        combined_fig.update_layout(template='plotly_dark')
        return combined_fig



class OLSStrategy(baseStrategy):
    def __init__(self, price_df, pairs, start_date, end_date, parameters=
    {'z_entry_threshold': 1.5, 'z_exit_threshold': 0, 'stop_loss_threshold': 2, 'z_restart_threshold': 0}):
        super().__init__(price_df, pairs, start_date, end_date, parameters)
        self.pair_model_attributes = {}

    def generate_model(self, pair, training_data):
        ticker1, ticker2 = pair

        training_data = training_data[[ticker1, ticker2]].dropna()

        # Implement OLS regression
        x = sm.add_constant(training_data[ticker1]).values
        y =training_data[ticker2].values

        model = sm.OLS(y, x).fit()
        alpha, beta = model.params

        training_data['spread'] = training_data[ticker2] - beta * training_data[ticker1] - alpha

        spread_mean = training_data.spread.mean()
        spread_std = training_data.spread.std()

        self.pair_model_attributes[tuple(pair)] = {'model': model, 'alpha': alpha, 'beta': beta , 'spread_mean': spread_mean , 'spread_std': spread_std}

    def generate_all_models(self , training_data):
        for pair in self.pairs:
            self.generate_model(pair = pair , training_data = training_data)

    def generate_signals(self, pair , predict_data):
        z_entry_threshold = self.parameters['z_entry_threshold']
        z_exit_threshold = self.parameters['z_exit_threshold']
        stop_loss_threshold = self.parameters['stop_loss_threshold']
        z_restart_threshold = self.parameters['z_restart_threshold']

        ticker1, ticker2 = pair

        alpha = self.pair_model_attributes[tuple(pair)]['alpha']
        beta = self.pair_model_attributes[tuple(pair)]['beta']
        spread_mean = self.pair_model_attributes[tuple(pair)]['spread_mean']
        spread_std = self.pair_model_attributes[tuple(pair)]['spread_std']


        data = predict_data[[ticker1, ticker2]].dropna()
        data['spread'] = data[ticker2] - beta * data[ticker1] - alpha

        z_score = (data.spread - spread_mean) / spread_std

        # Initialise series to hold signals
        signals = pd.Series(index=z_score.index).fillna(0)


        # Initialize variables to hold state
        position = 0
        stop_loss_triggered = False
        stop_trigger_side = 0

        # Loop through each z-score
        # If z_score outside entry_threshold, enter position
        # If position is open and z_score outside stop_loss_threshold, close position
        # If position is open and z_score crosses exit_threshold, close position
        # If stop loss is triggered, wait for z_score to cross restart_threshold before re-entering position

        for i in range(len(z_score)):
            if stop_loss_triggered:
                signals[i] = 0  # Signal to close position

                # Check if we can reset stop-loss trigger
                if z_score[i]*stop_trigger_side*-1 <= z_restart_threshold:
                    stop_loss_triggered = False  # Reset stop-loss trigger

            if not stop_loss_triggered:
                if position == 0:  # No position
                    if z_score[i] > z_entry_threshold:
                        position = -1  # Short
                        signals[i] = -1
                    elif z_score[i] < -z_entry_threshold:
                        position = 1  # Long
                        signals[i] = 1

                elif position == 1:  # Long on the spread
                    if z_score[i] > -z_exit_threshold:
                        position = 0  # Close position
                        signals[i] = -1
                    elif z_score[i] < -stop_loss_threshold:
                        position = 0  # Close position
                        signals[i] = -1
                        stop_trigger_side = 1
                        stop_loss_triggered = True  # Activate stop-loss

                elif position == -1:  # Short on the spread
                    if z_score[i] < z_exit_threshold:
                        position = 0  # Close position
                        signals[i] = 1
                    elif z_score[i] > stop_loss_threshold:
                        position = 0  # Close position
                        signals[i] = 1
                        stop_trigger_side = -1
                        stop_loss_triggered = True  # Activate stop-loss

            # Forward fill the signals
        signals = signals.fillna(method='ffill').fillna(0)
        return signals , z_score , beta

    def backtest_pair(self, pair, backtest_data):
        signals , z_score , beta = self.generate_signals(pair , predict_data = backtest_data)
        ticker1 , ticker2 = pair

        prices = backtest_data[[ticker1, ticker2]].dropna()



        # Create portfolio DataFrame to store values
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['spread_price'] = prices[ticker2] - beta*prices[ticker1]
        portfolio['signals'] = signals
        portfolio['z_score'] = z_score
        portfolio['position'] = portfolio['signals'].cumsum() # Buy=1, Sell=-1, Hold=0
        portfolio['price_delta'] = portfolio['spread_price'].diff().fillna(0)

        portfolio['profit'] = (portfolio['price_delta'] * portfolio['position'].shift(1)).fillna(0)
        portfolio['equity_curve'] = portfolio['profit'].cumsum()


        return portfolio




class KalmanStrategy(baseStrategy):
    def __init__(self):
        super().__init__()


class distanceStrategy(baseStrategy):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    from scripts.db.db_interface import dbInterface
    from scripts.utils.util_functions import read_config
    from scripts.pairs_selection.pairs_selector import cointergrationSelector

    config = read_config()
    tickers_collection_name = config['database']['collections']['tickers_collection_name']
    price_collection_name = config['database']['collections']['price_collection_name']
    batch_size = config['database']['batch_size']
    lookback_days = 200
    start_data = dt.date.today() - dt.timedelta(days=lookback_days)
    end_data = config['data']['end_data']

    db = dbInterface(config=config['database'], ticker_collection=tickers_collection_name,
                     price_collection=price_collection_name)
    tickers = db.retrieve_tickers(sectors = ['Energy'] , indices = ['S&P 500'])
    prices = db.retrieve_prices(tickers = tickers)[
        ['adj close']].reset_index()
    prices = prices.pivot_table(index='date', columns='ticker', values='adj close')

    selector = cointergrationSelector(prices, time_period=lookback_days, top_n=50)

    closest_pairs = selector.get_pairs()
    closest_pair = closest_pairs[0]

    strategy = OLSStrategy(price_df=prices, pairs=closest_pairs, start_date=start_data, end_date=end_data)

    train_test_split = 0.6
    train_test_split_index = int(len(strategy.price_df) * train_test_split)
    training_data = strategy.price_df.iloc[:train_test_split_index]
    backtest_data = strategy.price_df.iloc[train_test_split_index:]

    strategy.generate_all_models(training_data=training_data)
    strategy.backtest_portfolio(backtest_data= backtest_data)

