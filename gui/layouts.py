from dash import html, dcc
import dash_bootstrap_components as dbc

layout = html.Div([
    dcc.Tabs(id="tabs", value='pair-selection', children=[
        dcc.Tab(label='Pair Selection', value='pair-selection'),
        dcc.Tab(label='Strategy Backtest', value='strategy-backtest'),
        ],
        className='mb-4'
    ),
    html.Div(id='tabs-content'),

])