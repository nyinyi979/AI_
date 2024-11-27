import dash
from dash import html, dcc, callback, Input, Output

dash.register_page(__name__,title="Analysis",path="/analysis")

layout = html.Div([
    html.H1('This is our Analytics page'),
])