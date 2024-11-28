import dash
import base64
import io
import pandas as pd
from components.Button import Button
from components.Typography import P
from dash import html, dcc, callback, Input, Output, State

dash.register_page(__name__, path="/")
layout = html.Div(
    children=[
        dcc.Store(id="file-store", storage_type="local"),
        html.Div(
            children=[
                P(
                    "Find the details on your model, dataset in one click.",
                    variant="heading1",
                    className="text-center",
                ),
                dcc.Upload(
                    children=[
                        P(
                            "Upload CSV file to get started...",
                            variant="body2",
                            className="text-[#424242] ",
                            id="filename"
                        ),
                        Button(
                            children=[
                                "Upload",
                                html.Img(src="/assets/images/upload.svg"),
                            ],
                            variant="primary",
                            size="md",
                            className="flex items-center gap-2.5",
                        ),
                    ],
                    accept=".csv",
                    id="file-upload",
                    className="w-[826px] block group flex items-center justify-between mx-auto p-3 rounded-xl bg-[#D2E9E9] cursor-pointer",
                ),
                Button(
                    children="Start Analysing",
                    id="start-analyse",
                    disabled=True,
                    variant="primary",
                    size="md",
                    className="flex items-center gap-2.5",
                ),
            ],
            className="max-w-[858px] relative flex flex-col gap-5 items-center justify-center mt-40 z-[10] mx-auto bg-transparent",
        ),
        html.Img(
            src="assets/images/bg-box.png",
            className="absolute -left-[5%] top-[158px] opacity-[80%] z-[2]",
        ),
        html.Img(
            src="assets/images/bg-box.png",
            className="absolute -left-[15%] top-[457px] opacity-[80%] z-[1]",
        ),
        html.Img(
            src="assets/images/bg-box.png",
            className="absolute -right-[20%] top-[100px] z-[1]",
        ),
        html.Img(
            src="assets/images/bg-box.png",
            className="absolute -right-[15%] top-[400px] z-[1]",
        ),
    ],
    className="relative w-full",
)


@callback(
    Output("filename", "children"),
    Input("file-upload", "contents"),
    State("file-upload", "filename"),
    prevent_initial_call=True,
)
def handleFileUpload(contents, filename):
    return filename


@callback(
    Output("file-store", "data"),
    Output("start-analyse", "disabled"),
    Input("file-upload", "contents"),
    prevent_initial_call=True,
)
def storeData(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    return content_string, False
