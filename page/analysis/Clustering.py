from dash import html, dcc
from components.Typography import P
import plotly.graph_objs as go


def Clustering():
    return html.Div(
        [
            P(
                "K-Means Clustering",
                variant="heading2",
                className="mb-4",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            P(
                                "Unlabeled Data",
                                variant="body1",
                                className="mb-2",
                            ),
                            dcc.Graph(
                                id="unlabeled-data",
                                figure={
                                    "layout": go.Layout(
                                        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
                                        paper_bgcolor="rgba(0,0,0,0)",  # Transparent surrounding area
                                    )
                                },
                                config={
                                    "displayModeBar": False,  # Show mode bar
                                    "displaylogo": False,  # Hide Plotly logo
                                    "modeBarButtonsToRemove": [
                                        "toImage" "zoom"
                                    ],  # Optional: Remove specific buttons
                                },
                            ),
                        ],
                        className="mb-4",
                    ),
                    html.Div(
                        [
                            P(
                                "Clustered Data",
                                variant="body1",
                                className="mb-2",
                            ),
                            dcc.Graph(id="clustered-data"),
                        ],
                        className="mb-4",
                    ),
                ],
                className="grid grid-cols-2",
            ),
        ],
        className="p-4",
    )
