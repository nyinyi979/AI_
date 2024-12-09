from dash import html, dcc, callback, Input, Output
from components.Typography import P


def ClassifierDialog():
    return html.Div(
        html.Div(
            children=[
                html.Label("Select Classifier:"),
                dcc.Dropdown(
                    id="classifier-dropdown",
                    options=[
                        {"label": "AdaBoost Classifier", "value": "adaboost"},
                        {"label": "Random Forest Classifier", "value": "randomforest"},
                    ],
                    value="adaboost",  # Default value is AdaBoost
                    clearable=False,
                    style={"width": "100%"},  # Larger dropdown width
                ),
                html.Label("Select Features (x):"),
                dcc.Dropdown(
                    id="x-columns",
                    multi=True,
                    placeholder="Select feature columns",
                ),
                html.Label("Select Target (y):"),
                dcc.Dropdown(
                    id="y-columns",
                    multi=False,
                    placeholder="Select target column",
                ),
                html.Label("Select Test Size:"),
                dcc.Slider(
                    id="test-size-slider",
                    min=0.1,
                    max=0.9,
                    step=0.1,
                    value=0.3,
                    marks={
                        i: f"{i:.1f}"
                        for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    },
                ),
                html.Label("Select Train Size:"),
                dcc.Slider(
                    id="train-size-slider",
                    min=0.1,
                    max=0.9,
                    step=0.1,
                    value=0.7,
                    marks={
                        i: f"{i:.1f}"
                        for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    },
                ),
                html.Button(
                    id="classifier-close-btn",
                    children=[
                        html.Img(src="assets/images/cross.svg", className="size-6")
                    ],
                    className="absolute right-3 top-3",
                ),
            ],
            className="w-[814px] h-[400px] overflow-auto flex flex-col gap-4 fixed left-[50%] top-[50%] -translate-x-[50%] -translate-y-[50%] p-5 rounded-xl bg-[#D2E9E9] duration-300",
            style={"boxShadow": "0 0 30px 0px rgba(0, 0, 0, 0.50)"},
        ),
        id="classifier-dialog",
        className="w-full h-full hidden fixed left-0 top-0 z-[120] bg-black/20",
        style={"display": "none"},
    )


@callback(
    Output("classifier-dialog", "style", allow_duplicate=True),
    Input("classifier-close-btn", "n_clicks"),
    prevent_initial_call=True,
)
def closeDialog(n_clicks):
    return {"display": "none"}


@callback(
    Output("classifier-dialog", "style", allow_duplicate=True),
    Input("Classifier-setting", "n_clicks"),
    prevent_initial_call=True,
)
def openDataDialog(n_clicks):
    if n_clicks:
        return {"boxShadow": "0 0 30px 0px rgba(0, 0, 0, 0.50)", "display": "block"}
    else:
        return None
