from dash import html, callback, Input, Output, State, dcc
from components.Typography import P
from components.Button import Button

import dash_daq as daq
import pandas as pd


def DataDialog():
    return html.Dialog(
        id="dialog_id",
        children=[
            dcc.Store(id="used_col_row", storage_type="local"),
            P("Dialog Name", variant="body1"),
            html.P(id="form-output"),
            html.Button(id="close_btn"),
            daq.ToggleSwitch(id="use_row", value=False),
            dcc.Checklist(id="checklist"),
            html.P(id="checked"),
        ],
        className="w-fit h-fit flex flex-col gap-3 absolute left-[50%] top-[50%] -translate-x-[50%] -translate-y-[50%] bg-[#D2E9E9] z-[20]",
        style={"boxShadow": "0 0 30px 0px rgba(0, 0, 0, 0.50)", "display": "none"},
    )


@callback(
    Output("dialog_id", "style", allow_duplicate=True),
    Input("close_btn", "n_clicks"),
    prevent_initial_call=True,
)
def closeDialog(n_clicks):
    return {"display": "none"}


@callback(
    Output("checklist", "options"),
    Input("file-store", "data"),
    prevent_initial_call=True,
)
def process_form(file):
    if file and file["content"]:
        df = pd.DataFrame(file["content"])
        children = []
        for col in df.columns:
            children.append(col)
        return children
    else:
        return None


@callback(
    Output("used_col_row", "data"),
    Input("checklist", "value"),
    Input("use_row", "value"),
)
def updateColRow(checkedValues, use_row):
    if use_row:
        return {"use_row": True, "values": checkedValues}
    else:
        return {"use_row": False, "values": checkedValues}
