import dash
from page.Home import Home

dash.register_page(__name__, path="/")

layout = Home()
