from dash import html, dcc, callback, Input, Output
from components.Typography import P


def ClusteringDialog():
    return html.Div(
        html.Div(
            children=[
                P("Clustering Controls", variant="body1"),
                html.Div(
                    [
                        # Number of Clusters Slider
                        html.Div(
                            [
                                P(
                                    "Number of Clusters",
                                    variant="body2",
                                    className="mb-2",
                                ),
                                dcc.Slider(
                                    id="n-clusters-slider",
                                    min=2,
                                    max=10,
                                    value=3,
                                    marks={i: str(i) for i in range(2, 11)},
                                    step=1,
                                ),
                            ],
                            className="mb-4",
                        ),
                        # Scaling Method Dropdown
                        html.Div(
                            [
                                P("Scaling Method", variant="body2", className="mb-2"),
                                dcc.Dropdown(
                                    id="scaling-method-dropdown",
                                    options=[
                                        {
                                            "label": "Standard Scaling",
                                            "value": "standard",
                                        },
                                        {"label": "Min-Max Scaling", "value": "minmax"},
                                    ],
                                    value="standard",
                                    clearable=False,
                                ),
                            ],
                            className="mb-4",
                        ),
                    ],
                    className="p-4 bg-[#eeffff] rounded-lg",
                ),
                html.Button(
                    id="clustering-close-btn",
                    children=[
                        html.Img(src="assets/images/cross.svg", className="size-6")
                    ],
                    className="absolute right-3 top-3",
                ),
                # Cluster Summary
                html.Div(
                    id="cluster-summary",
                    className="p-4 bg-[#eeffff] rounded-lg",
                ),
            ],
            className="w-[814px] h-[400px] overflow-auto flex flex-col gap-4 fixed left-[50%] top-[50%] -translate-x-[50%] -translate-y-[50%] p-5 rounded-xl bg-[#D2E9E9] duration-300",
            style={"boxShadow": "0 0 30px 0px rgba(0, 0, 0, 0.50)"},
        ),
        id="clustering-dialog",
        className="w-full h-full hidden fixed left-0 top-0 z-[120] bg-black/20",
        style={"display": "none"},
    )


@callback(
    Output("clustering-dialog", "style", allow_duplicate=True),
    Input("clustering-close-btn", "n_clicks"),
    prevent_initial_call=True,
)
def closeDialog(n_clicks):
    return {"display": "none"}