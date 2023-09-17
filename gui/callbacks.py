from scripts.utils.util_functions import read_config
from scripts.pairs_selection.pairs_selector import distanceSelector, correlationSelector, cointergrationSelector
from scripts.strategies.strategy import OLSStrategy, KalmanStrategy
from scripts.utils.util_classes import CustomLogger

from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
from dash import callback_context
import datetime as dt
import plotly.graph_objects as go
import pandas as pd

logger = CustomLogger.get_logger(name=__name__)

def render_content(tab, unique_sectors, unique_indices):
    config = read_config(key='data')
    min_date = config['start_data']
    max_date = dt.date.today()
    if tab == 'pair-selection':
        return html.Div([
            dcc.Store(id='selected-pairs'),
            dbc.Container(
                [
                    dbc.Row([
                        dbc.Col([
                            dbc.DropdownMenu(
                                [
                                    dbc.DropdownMenuItem("Cointegration", id="cointegration-method"),
                                    dbc.DropdownMenuItem("Distance", id="distance-method"),
                                    dbc.DropdownMenuItem("Correlation", id="correlation-method"),
                                ],
                                label="Select Pair Selection Method",
                                id="pairselector-dropdown",
                                className='mb-4 d-flex justify-content-center align-items-center'
                            )
                        ]),
                        dbc.Col([
                            dbc.DropdownMenu(
                                [dbc.DropdownMenuItem(sector, id=f"sector-{sector}") for sector in unique_sectors],
                                label="Select Sector",
                                id="sector-dropdown",
                                className='mb-4 d-flex justify-content-center align-items-center'
                            )
                        ]),
                        dbc.Col([
                            dbc.DropdownMenu(
                                [dbc.DropdownMenuItem(index, id=f"index-{index}") for index in unique_indices],
                                label="Select Index",
                                id="index-dropdown",
                                className='mb-4 d-flex justify-content-center align-items-center'
                            )
                        ]),
                    ]),
                    dbc.Row([
                        dcc.Loading(
                            id="loading",
                            type="circle",  # or "default", "cube", "dot", "circular"
                            children=[
                                dcc.Graph(id='pair-selection-graph', className='mb-4'),
                            ]
                        )
                    ]),

                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(html.Div('Select Date Range'), body=True, color="dark", outline=True
                                         , style={'height': '100%', 'color': 'white'}, className='btn-primary'),
                                width={'size': 4},
                                className='d-flex justify-content-center align-items-center'
                            ),
                            dbc.Col(
                                dcc.DatePickerRange(
                                    id='date-picker',
                                    min_date_allowed=min_date,
                                    max_date_allowed=max_date,
                                    initial_visible_month=max_date,
                                    start_date=max_date - dt.timedelta(days=200),
                                    end_date=max_date,
                                ),
                                width={"size": 4},
                                className='d-flex justify-content-center align-items-center'
                            ),
                            dbc.Col(
                                dbc.Card(
                                    html.Div(id='output-days-box', children=f"Default days: {200}",
                                             style={'color': 'white'}),
                                    body=True, color="dark", outline=True, className='btn-primary'
                                ),
                                width={"size": 4},
                                className='d-flex justify-content-center align-items-center'
                            ),
                        ],
                        className='mb-4'
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button('Run Selection Method', id='run-button', n_clicks=0,
                                       style={"height": "60px", "width": "100%"})
                        ], width={"size": 8}, style={'margin-bottom': '40px'})
                    ],
                        className='mb-4',
                        justify='center'
                    ),

                    dbc.Row([
                        dbc.Col([
                            dbc.Input(id='name-tag-input', type="text", style={"height": "60px", "width": "100%"})
                        ], width={"size": 4}),
                        dbc.Col([
                            dbc.Button('Save Pairs', id='save-button', n_clicks=0,
                                       style={"height": "60px", "width": "100%"})
                        ], width={"size": 4}),
                        dbc.Col([
                            html.Div(id='save-output', style={"width": "100%"})
                        ], width={"size": 4})
                    ],
                        className='mb-4'
                    ),
                    dbc.Row([
                        dbc.Card(html.H3('Selected Pairs', style={'textAlign': 'center', 'color': 'white'}), body=True,
                                 color="dark", outline=True, className='btn-primary')
                    ]),
                    dbc.Row([
                        dbc.Table.from_dataframe(pd.DataFrame(columns=['ticker 1', 'ticker 2']), id='pairs-table'
                                                 , striped=True, bordered=True, hover=True,
                                                 style={'textAlign': 'center'})
                    ])
                ]
            )

        ])
    elif tab == 'strategy-backtest':
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Button('Load Pairs', id='load-pairs-button'),
                ]),
                dbc.Col([
                    dcc.Dropdown(id='nametag-dropdown')
                ]),
                dbc.Col([
                    dbc.DropdownMenu(
                        [
                            dbc.DropdownMenuItem("OLS Strategy", id="ols-strategy"),
                            dbc.DropdownMenuItem("Kalman Strategy", id="kalman-strategy"),
                        ],
                        label="Select Strategy type",
                        id="strategy-dropdown",
                        className='mb-4 d-flex justify-content-center align-items-center'
                    )
                ])

            ],
                className='mb-4'
            ),
            dbc.Row([
                dbc.Col(
                    dcc.DatePickerRange(
                        id='strategydate-picker',
                        min_date_allowed=min_date,
                        max_date_allowed=max_date,
                        initial_visible_month=max_date,
                        start_date=max_date - dt.timedelta(days=200),
                        end_date=max_date,
                    ),
                    width={"size": 4},
                    className='d-flex justify-content-center align-items-center'
                )
            ],
                className='mb-4'
            ),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading",
                        type="circle",  # or "default", "cube", "dot", "circular"
                        children=[
                            dcc.Graph(id='portfolio-graph', className='mb-4'),
                        ]
                    )
                ], width = 8),
                dbc.Col([
                    dbc.Table([
                        html.Thead([
                            html.Tr([html.Th("Statistic"), html.Th("Value")])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("Max Drawdown"), html.Td(id='max_drawdown')]),
                            html.Tr([html.Td("Sharpe Ratio"), html.Td(id='sharpe_ratio')]),
                            html.Tr([html.Td("Sortino Ratio"), html.Td(id='sortino_ratio')]),
                        ])
                    ]),
                ], width=4),
            ]),

            dbc.Row([
                dbc.Col([
                    html.Label("Z-score trade entry threshold"),
                    dcc.Slider(
                        id='z-entry-slider',
                        min=0,
                        max=3,
                        value=1.5,
                        step=0.1,
                    ),
                ],
                    className='mb-4'
                ),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Z-score trade exit threshold"),
                    dcc.Slider(
                        id='z-exit-slider',
                        min=0,
                        max=3,
                        value=0,
                        step=0.1,
                    ),
                ],
                    className='mb-4'
                ),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Z-score stop-loss threshold"),
                    dcc.Slider(
                        id='z-stoploss-slider',
                        min=0,
                        max=3,
                        value=2,
                        step=0.1,
                    ),
                ],
                    className='mb-4'
                ),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Z-score rentry after stop-loss threshold"),
                    dcc.Slider(
                        id='z-reentry-slider',
                        min=0,
                        max=3,
                        value=0,
                        step=0.1,
                    ),
                ],
                    className='mb-4'
                ),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card(html.H2('Individual Pair Detail'), body=True, color="dark", outline=True , className='btn-primary',
                             style={"width": "50%", "text-align": "center" , 'color' : 'white'})
                ], width={"size": 12}, className='mb-4 d-flex justify-content-center align-items-center')
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='pairs-dropdown',
                        placeholder='Select pair'  # Placeholder text
                    )  # This dropdown's options will be updated dynamically
                ], width={"size": 4})
            ], className='mb-4 d-flex justify-content-center align-items-center'),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(dcc.Graph(id='pair-graph',
                                          style={
                                              'height': '800px',
                                              'width': '800px',
                                              'margin-left': 'auto',
                                              'margin-right': 'auto'
                                          }))
                ])
            ], className='justify-content-center align-items-center')
        ] )


def register_callbacks(app, db):
    unique_sectors = db.retrive_sectors() + ['All']
    unique_indices = db.retrieve_indices() + ['All']

    @app.callback(
        Output('tabs-content', 'children'),
        [Input('tabs', 'value')]
    )
    def update_content(tab):
        return render_content(tab, unique_sectors, unique_indices)

    @app.callback(
        Output('save-output', 'children'),
        [Input('save-button', 'n_clicks'),
         ],
        [
            State('selected-pairs', 'data'),
            State('name-tag-input', 'value')]
    )
    def save_to_db(n_clicks, pairs, name_tag):
        if n_clicks is None:
            return "No clicks yet"
        elif not name_tag:
            return dbc.Alert("Enter a name tag to insert selection to database", color="danger")
        else:
            if pairs is None:
                return dbc.Alert("No pairs selected", color="danger")
            else:
                # Create df with name_tag and pairs_list as column names
                df = pd.DataFrame(columns=['nametag', 'pairs'])
                df.loc[0] = [name_tag, pairs]
                logger.info('Saving pairs to db with nametag : {}'.format(name_tag))
                db.write_dataframe(df, 'pairsCollection')
                return dbc.Alert(f"Saved pairs with tag : {name_tag} to db", color="success")

    @app.callback(
        Output('output-days-box', 'children'),
        [Input('date-picker', 'start_date'),
         Input('date-picker', 'end_date')]
    )
    def update_date_box(start_date, end_date):
        start = dt.datetime.strptime(start_date.split(" ")[0], "%Y-%m-%d")
        end = dt.datetime.strptime(end_date.split(" ")[0], "%Y-%m-%d")
        days = (end - start).days
        return f"Selected range: {days} days"

    @app.callback(
        Output('pairselector-dropdown', 'label'),
        [Input('distance-method', 'n_clicks'),
         Input('correlation-method', 'n_clicks'),
         Input('cointegration-method', 'n_clicks')
         ]
    )
    def update_card(n1, n2, n3):
        ctx = callback_context
        if not ctx.triggered:
            return "Select Pair Selection Method"

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        method = "Select Pair Selection Method"

        if trigger_id == 'distance-method':
            method = 'Distance'
        elif trigger_id == 'correlation-method':
            method = 'Correlation'
        elif trigger_id == 'cointegration-method':
            method = 'Cointegration'

        return method

    @app.callback(
        Output('strategy-dropdown', 'label'),
        [Input('ols-strategy', 'n_clicks'),
         Input('kalman-strategy', 'n_clicks'),
         ]
    )
    def update_strategy_card(n1, n2):
        ctx = callback_context
        if not ctx.triggered:
            return "Select Strategy type"

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        method = "Select Pair Selection Method"

        if trigger_id == 'ols-strategy':
            method = 'OLS Strategy'
        elif trigger_id == 'kalman-strategy':
            method = 'Kalman Strategy'

        return method

    @app.callback(
        Output("sector-dropdown", "label"),
        [Input(f"sector-{sector}", "n_clicks") for sector in unique_sectors],
        [State("sector-dropdown", "label")],
        prevent_initial_call=True
    )
    def update_sector_label(*args):
        ctx = callback_context
        if not ctx.triggered:
            return "Select Sector"
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            return button_id.split('-')[1]

    @app.callback(
        Output("index-dropdown", "label"),
        [Input(f"index-{index}", "n_clicks") for index in unique_indices],
        [State("index-dropdown", "label")],
        prevent_initial_call=True
    )
    def update_index_label(*args):
        ctx = callback_context
        if not ctx.triggered:
            return "Select Index"
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            return button_id.split('-')[1]

    @app.callback(
        [Output("pair-selection-graph", "figure"),
         Output("selected-pairs", "data")],
        Input('run-button', 'n_clicks'),
        State("date-picker", "start_date"),
        State("date-picker", "end_date"),
        State("pairselector-dropdown", "label"),
        State("sector-dropdown", "label"),
        State("index-dropdown", "label")
    )
    def update_graph(n_clicks, start_date, end_date, method, sector, index):
        if n_clicks is None or sector not in unique_sectors or index not in unique_indices or method == 'Select Pair Selection Method':
            fig = go.Figure()

            # Apply the dark mode template
            fig.update_layout(template='plotly_dark')
            return fig, None

        if sector == 'All':
            sectors = None
        else:
            sectors = [sector]

        if index == 'All':
            indices = None
        else:
            indices = [index]

        tickers = db.retrieve_tickers(sectors=sectors, indices=indices)
        prices = db.retrieve_prices(tickers=tickers, start_date=start_date, end_date=end_date)
        prices = prices.pivot_table(index='date', columns='ticker', values='adj close')

        if method == 'Distance':
            pairs_selector = distanceSelector(prices, time_period=9999, top_n=30)
        elif method == 'Correlation':
            pairs_selector = correlationSelector(prices, time_period=9999, top_n=30)
        elif method == 'Cointegration':
            pairs_selector = cointergrationSelector(prices, time_period=9999, top_n=30)

        pairs = pairs_selector.get_pairs()
        fig = pairs_selector.plot_summary(pairs=pairs)

        return fig, pairs

    @app.callback(
        Output('nametag-dropdown', 'options'),
        Input('load-pairs-button', 'n_clicks')
    )
    def update_dropdown_options(n_clicks):
        if n_clicks is None:
            return []  # No options if the button hasn't been clicked

        return db.retrieve_saved_pair_names()

    @app.callback(
        [Output('portfolio-graph', 'figure'),
         Output('max_drawdown', 'children'),
         Output('sharpe_ratio', 'children'),
         Output('sortino_ratio', 'children')],
        [
            Input('nametag-dropdown', 'value'),
            Input('strategydate-picker', 'start_date'),
            Input('strategydate-picker', 'end_date'),
            Input('z-entry-slider', 'value'),
            Input('z-exit-slider', 'value'),
            Input('z-stoploss-slider', 'value'),
            Input('z-reentry-slider', 'value'),
            Input('strategy-dropdown', 'label')
        ]
    )
    def update_portfolio_graph(nametag, start_date, end_date, z_entry, z_exit, z_stop, z_reentry, strategy):
        if nametag is None or strategy is None or strategy == 'Select Strategy type':
            fig = go.Figure()
            fig.update_layout(template='plotly_dark')
            return fig , None , None , None

        pairs = db.retrieve_pairs(nametag)
        tickers = list(set([ticker for pair in pairs for ticker in pair]))
        prices = db.retrieve_prices(tickers=tickers, start_date=start_date, end_date=end_date)
        prices = prices.pivot_table(index='date', columns='ticker', values='adj close')

        if strategy == 'OLS Strategy':
            strategy = OLSStrategy(price_df=prices, pairs=pairs, start_date=start_date, end_date=end_date
                                   , parameters={'z_entry_threshold': z_entry, 'z_exit_threshold': z_exit,
                                                 'stop_loss_threshold': z_stop, 'z_restart_threshold': z_reentry})
        elif strategy == 'Kalman Strategy':
            strategy = KalmanStrategy()

        # Use same data for training and backtest , some lookahead bias here. Should really be using different windows
        # for training and backtest.
        training_data = prices
        backtest_data = prices

        strategy.generate_all_models(training_data=training_data)
        strategy.backtest_portfolio(backtest_data=backtest_data)

        return (strategy.plot_portfolio_statistics() , strategy.portfolio_statistics['max_drawdown']
                , strategy.portfolio_statistics['sharpe_ratio'] , strategy.portfolio_statistics['sortino_ratio'])

    @app.callback(
        Output("pairs-table", "children"),
        Input("selected-pairs", "data")
    )
    def populate_table(selected_pairs_data):
        if selected_pairs_data is None:
            return dbc.Table.from_dataframe(pd.DataFrame(columns=['ticker 1', 'ticker 2']), id='pair-table'
                                            , striped=True, bordered=True, hover=True, style={'textAlign': 'center'})

        # Convert the data to a DataFrame
        df = pd.DataFrame(selected_pairs_data, columns=['ticker1', 'ticker2'])
        return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, style={'textAlign': 'center'})

    @app.callback(
        Output('pairs-dropdown', 'options'),
        Input('nametag-dropdown', 'value'),
    )
    def update_dropdown_options(selected_nametag):
        if not selected_nametag:
            return []

        # Fetch pairs related to the selected nametag from the database
        pairs = db.retrieve_pairs(nametag = selected_nametag)
        options = [{"label": str(tuple(pair)), "value": str(tuple(pair))} for pair in pairs]

        return options

    @app.callback(
        Output('pair-graph', 'figure'),
        Input('pairs-dropdown', 'value'),
        Input('strategy-dropdown', 'label'),
        Input('strategydate-picker', 'start_date'),
        Input('strategydate-picker', 'end_date'),
        Input('z-entry-slider', 'value'),
        Input('z-exit-slider', 'value'),
        Input('z-stoploss-slider', 'value'),
        Input('z-reentry-slider', 'value')
    )
    def update_graph(selected_pair , strategy , start_date , end_date , z_entry , z_exit , z_stop , z_reentry):
        if selected_pair is None or strategy is None or selected_pair == 'Select pair':
            fig = go.Figure()
            fig.update_layout(template='plotly_dark')
            return fig

        # Extract pair from ('ticker1' , 'ticker2') format
        pair = eval(selected_pair)
        # Convert tuple to list
        pair= list(pair)


        pairs = [pair]



        prices = db.retrieve_prices(tickers=pair, start_date=start_date, end_date=end_date)
        prices = prices.pivot_table(index='date', columns='ticker', values='adj close')

        if strategy == 'OLS Strategy':
            strategy = OLSStrategy(price_df=prices, pairs=pairs, start_date=start_date, end_date=end_date
                                   , parameters={'z_entry_threshold': z_entry, 'z_exit_threshold': z_exit,
                                                 'stop_loss_threshold': z_stop, 'z_restart_threshold': z_reentry})
        elif strategy == 'Kalman Strategy':
            strategy = KalmanStrategy()

        strategy.generate_model(pair = pair , training_data=prices)

        return strategy.plot_pair(pair=pair)
