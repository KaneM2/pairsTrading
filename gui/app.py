from gui.layouts import layout
from gui.callbacks import register_callbacks

from dash import Dash
import dash_bootstrap_components as dbc





app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = layout


def run_app(db):
    # Register callbacks
    register_callbacks(app , db)
    app.run_server(debug=True)

if __name__ == "__main__":
    run_app()