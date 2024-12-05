from dash import html
from page.home.Layout import Layout


def Home():
    return html.Div(
        [Layout()],
        className="relative",
    )
