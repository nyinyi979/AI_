import dash
from page.Analysis import Analysis

dash.register_page(__name__, path="/analysis")

layout = Analysis()
