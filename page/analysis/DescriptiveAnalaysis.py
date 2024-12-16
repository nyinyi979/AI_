from dash import html, dcc, callback, Input, Output
from components.Button import Button
from components.Typography import P
from dash_svg import Svg, Path

import plotly.graph_objs as go
import pandas as pd
import numpy as np


def DescriptiveAnalysis():
    return html.Div(
        [
            dcc.Store(id="file-store", storage_type="local"),
            dcc.Store(id="used-col-row", storage_type="local"),
            html.Div(
                [
                    Button(
                        children=[
                            "File name.csv",
                            html.Img(src="assets/images/edit.svg", className="size-6"),
                        ],
                        variant="primary",
                        id="uploaded-filename",
                        n_clicks=0,
                        className="w-fit flex items-center gap-2.5 mb-4 rounded-[5px]",
                    ),
                    # Descriptive Analysis
                    P([], variant="body1", className="my-0.5", id="title"),
                    html.Div(
                        [
                            html.Button(
                                [
                                    P(
                                        "Mean value",
                                        variant="body2",
                                    ),
                                    Svg(
                                        [
                                            Path(
                                                d="M9 10L12.2581 12.4436C12.6766 12.7574 13.2662 12.6957 13.6107 12.3021L20 5",
                                                strokeWidth="2",
                                                strokeLinecap="round",
                                                className="stroke-[#C4DFDF] group-hover:stroke-[#141717] duration-300",
                                            ),
                                            Path(
                                                d="M21 12C21 13.8805 20.411 15.7137 19.3156 17.2423C18.2203 18.7709 16.6736 19.9179 14.893 20.5224C13.1123 21.1268 11.187 21.1583 9.38744 20.6125C7.58792 20.0666 6.00459 18.9707 4.85982 17.4789C3.71505 15.987 3.06635 14.174 3.00482 12.2945C2.94329 10.415 3.47203 8.56344 4.51677 6.99987C5.56152 5.4363 7.06979 4.23925 8.82975 3.57685C10.5897 2.91444 12.513 2.81996 14.3294 3.30667",
                                                stroke="#141717",
                                                strokeWidth="2",
                                                className="stroke-[#C4DFDF] group-hover:stroke-[#141717] duration-300",
                                            ),
                                        ],
                                        width="24",
                                        height="24",
                                        viewBox="0 0 24 24",
                                        fill="none",
                                        id="mean-svg",
                                    ),
                                ],
                                id="mean",
                                n_clicks_timestamp=0,
                                className="w-[230px] flex justify-between items-center group",
                            ),
                            html.Button(
                                [
                                    P(
                                        "Median value",
                                        variant="body2",
                                    ),
                                    Svg(
                                        [
                                            Path(
                                                d="M9 10L12.2581 12.4436C12.6766 12.7574 13.2662 12.6957 13.6107 12.3021L20 5",
                                                strokeWidth="2",
                                                strokeLinecap="round",
                                                className="stroke-[#C4DFDF] group-hover:stroke-[#141717] duration-300",
                                            ),
                                            Path(
                                                d="M21 12C21 13.8805 20.411 15.7137 19.3156 17.2423C18.2203 18.7709 16.6736 19.9179 14.893 20.5224C13.1123 21.1268 11.187 21.1583 9.38744 20.6125C7.58792 20.0666 6.00459 18.9707 4.85982 17.4789C3.71505 15.987 3.06635 14.174 3.00482 12.2945C2.94329 10.415 3.47203 8.56344 4.51677 6.99987C5.56152 5.4363 7.06979 4.23925 8.82975 3.57685C10.5897 2.91444 12.513 2.81996 14.3294 3.30667",
                                                stroke="#141717",
                                                strokeWidth="2",
                                                className="stroke-[#C4DFDF] group-hover:stroke-[#141717] duration-300",
                                            ),
                                        ],
                                        width="24",
                                        height="24",
                                        viewBox="0 0 24 24",
                                        fill="none",
                                        id="median-svg",
                                    ),
                                ],
                                id="median",
                                n_clicks_timestamp=0,
                                className="w-[230px] flex justify-between items-center group",
                            ),
                            html.Button(
                                [
                                    P(
                                        "Mode value",
                                        variant="body2",
                                    ),
                                    Svg(
                                        [
                                            Path(
                                                d="M9 10L12.2581 12.4436C12.6766 12.7574 13.2662 12.6957 13.6107 12.3021L20 5",
                                                strokeWidth="2",
                                                strokeLinecap="round",
                                                className="stroke-[#C4DFDF] group-hover:stroke-[#141717] duration-300",
                                            ),
                                            Path(
                                                d="M21 12C21 13.8805 20.411 15.7137 19.3156 17.2423C18.2203 18.7709 16.6736 19.9179 14.893 20.5224C13.1123 21.1268 11.187 21.1583 9.38744 20.6125C7.58792 20.0666 6.00459 18.9707 4.85982 17.4789C3.71505 15.987 3.06635 14.174 3.00482 12.2945C2.94329 10.415 3.47203 8.56344 4.51677 6.99987C5.56152 5.4363 7.06979 4.23925 8.82975 3.57685C10.5897 2.91444 12.513 2.81996 14.3294 3.30667",
                                                stroke="#141717",
                                                strokeWidth="2",
                                                className="stroke-[#C4DFDF] group-hover:stroke-[#141717] duration-300",
                                            ),
                                        ],
                                        width="24",
                                        height="24",
                                        viewBox="0 0 24 24",
                                        fill="none",
                                        id="mode-svg",
                                    ),
                                ],
                                id="mode",
                                n_clicks_timestamp=0,
                                className="w-[230px] flex justify-between items-center group",
                            ),
                            html.Div(
                                [
                                    P(
                                        "Row Counts",
                                        variant="body2",
                                    ),
                                    P("", variant="body2", id="row-count"),
                                ],
                                className="w-[230px] flex justify-between items-center group",
                            ),
                            html.Div(
                                [
                                    P(
                                        "Column Counts",
                                        variant="body2",
                                    ),
                                    P("", variant="body2", id="column-count"),
                                ],
                                className="w-[230px] flex justify-between items-center group",
                            ),
                        ],
                        id="buttons",
                        className="flex flex-col gap-4",
                    ),
                ],
                className="flex flex-col gap-4",
            ),
            dcc.Graph(id="mean-graph", config={"displayModeBar": False}),
        ],
        className="grid grid-cols-2 gap-10 items-center px-4 pt-4 pb-8 relative border-b border-[#B1CBCB] 2xl:border-none",
    )


@callback(
    Output("uploaded-filename", "children"),
    Output("mean-graph", "figure"),
    Output("title", "children"),
    Output("mean-svg", "className"),
    Output("median-svg", "className"),
    Output("mode-svg", "className"),
    Output("row-count", "children"),
    Output("column-count", "children"),
    Input("file-store", "data"),
    Input("used-col-row", "data"),
    Input("mean", "n_clicks_timestamp"),
    Input("median", "n_clicks_timestamp"),
    Input("mode", "n_clicks_timestamp"),
)
def loadData(file, usedColRow, mean, median, mode):
    if int(mode) > int(mean) and int(mode) > int(mean):
        type = "mode"
    elif int(median) > int(mean) and int(median) > int(mode):
        type = "median"
    else:
        type = "mean"
    if file["fileName"] and file["content"]:
        df = pd.DataFrame(file["content"])
        title = ""
        clean_df = df.select_dtypes(include=np.number).dropna()
        data = []
        if usedColRow["useRow"] and usedColRow["useRow"] is True:
            title = "Descriptive Analytics(Using Rows)"
        else:
            title = "Descriptive Analytics(Using Columns)"
        if type == "mean":
            if usedColRow["useRow"] and usedColRow["useRow"] is True:
                # iloc indexes can be used to access rows
                data = clean_df.iloc[usedColRow["values"]].mean(axis=1)
            else:
                # selecting rows
                data = clean_df[usedColRow["values"]].mean()
        elif type == "median":
            if usedColRow["useRow"] and usedColRow["useRow"] is True:
                data = clean_df.iloc[usedColRow["values"]].median(axis=1)
            else:
                data = clean_df[usedColRow["values"]].mode()
        else:
            if usedColRow["useRow"] and usedColRow["useRow"] is True:
                data = clean_df.iloc[usedColRow["values"]].median(axis=1)
            else:
                data = clean_df[usedColRow["values"]].mode()

        x = clean_df.columns
        if usedColRow["label"] and usedColRow["useRow"]:
            x = df[usedColRow["label"]]
        y = data

        figure = go.Figure(
            data=[go.Bar(x=x, y=y, marker=dict(color="white"))],
            layout=go.Layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="#D2E9E9",
                height=300,
                autosize=True,
            ),
        )
        return (
            [
                file["fileName"],
                html.Img(src="assets/images/edit.svg", className="size-6"),
            ],
            figure,
            title,
            "active-svg" if type == "mean" else "",
            "active-svg" if type == "median" else "",
            "active-svg" if type == "mode" else "",
            clean_df.shape[0],
            clean_df.shape[1],
        )


@callback(
    Output("data-dialog", "style", allow_duplicate=True),
    Input("uploaded-filename", "n_clicks"),
    prevent_initial_call=True,
)
def openDataDialog(n_clicks):
    if n_clicks:
        return {"boxShadow": "0 0 30px 0px rgba(0, 0, 0, 0.50)", "display": "block"}
    else:
        return None
