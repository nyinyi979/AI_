from dash import html, callback, Input, Output, State, dcc
from components.Typography import P
from components.Button import Button

import dash_daq as daq
import pandas as pd


def DataDialog():
    return html.Div(
        html.Div(
            children=[
                dcc.Store(id="used-col-row", storage_type="local"),
                dcc.Store(id="file-store", storage_type="local"),
                html.Button(
                    id="data-close-btn",
                    children=[
                        html.Img(src="assets/images/cross.svg", className="size-6")
                    ],
                    className="absolute right-3 top-3",
                ),
                P("Analysis Setting", variant="body1"),
                html.P(id="form-output"),
                daq.ToggleSwitch(
                    id="use-row",
                    value=True,
                    label="Using Row",
                    size=40,
                    labelPosition="bottom",
                ),
                dcc.Dropdown(id="label", className="my-2"),
                dcc.Checklist(id="checklist"),
                html.P(id="checked"),
            ],
            className="w-[814px] h-[400px] overflow-auto flex flex-col gap-2 fixed left-[50%] top-[50%] -translate-x-[50%] -translate-y-[50%] p-5 rounded-xl bg-[#D2E9E9] z-[120] duration-300",
            style={"boxShadow": "0 0 30px 0px rgba(0, 0, 0, 0.50)"},
        ),
        id="data-dialog",
        className="w-full h-full hidden fixed left-0 top-0 z-[120] bg-black/20",
        style={"display": "none"},
    )


@callback(
    Output("data-dialog", "style", allow_duplicate=True),
    Input("data-close-btn", "n_clicks"),
    prevent_initial_call=True,
)
def closeDialog(n_clicks):
    return {"display": "none"}


@callback(
    Output("label", "options"),
    Input("file-store", "data"),
    prevent_initial_call=True,
)
def getColNames(file):
    if file and file["content"]:
        df = pd.DataFrame(file["content"])
        cols = []
        for col in df.columns:
            cols.append(col)
        return cols
    else:
        return []


@callback(
    Output("checklist", "options"),
    Output("checklist", "value"),
    Output("label", "style"),
    Output("label", "value"),
    Output("use-row", "label"),
    Input("file-store", "data"),
    Input("use-row", "value"),
    prevent_initial_call=True,
)
def process_form(file, useRow):
    checklists = []
    if file and file["content"]:
        df = pd.DataFrame(file["content"])
        if useRow is True:
            checklists = df.select_dtypes(include="number").dropna().axes[0].tolist()
        else:
            checklists = df.select_dtypes(include="number").dropna().columns.tolist()
        return (
            checklists,
            checklists,
            {"display": "block"} if useRow else {"display": "none"},
            df.columns[0],
            "Using Rows" if useRow else "Using Columns",
        )
    else:
        return checklists, checklists, {"display": "block"}, [], "Using Rows"


@callback(
    Output("used-col-row", "data"),
    Input("checklist", "value"),
    Input("use-row", "value"),
    Input("label", "value"),
)
def updateColRow(checkedValues, useRow, label):
    if useRow:
        return {"useRow": True, "values": checkedValues, "label": label}
    else:
        return {
            "useRow": False,
            "values": checkedValues,
            "label": label,
        }
