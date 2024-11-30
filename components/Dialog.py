from dash import html, callback, Input, Output, State, dcc
from components.Typography import P
from components.Button import Button


def Dialog():
    return html.Dialog(
        id="dialog_id",
        children=[
            P("Dialog Name", variant="body1"),
            html.P(id="form-output"),
            html.Form(
                children=[
                    dcc.Input(id="user-form-name"),
                    dcc.Input(id="user-form-email"),
                    dcc.Input(id="user-form-age"),
                    html.Div(
                        children=[
                            Button("Submit", type="button", id="submit", n_clicks=0),
                            Button("Close", n_clicks=0, type="button", id="close_btn"),
                        ],
                        className="w-fit flex gap-3 mx-auto",
                    ),
                    html.Button(
                        html.Img(
                            src="assets/images/cross.svg",
                            className="size-6 absolute right-2 top-2",
                        ),
                        id="close_btn",
                        n_clicks=0,
                        type="button",
                    ),
                ],
                className="flex flex-col gap-2",
                method="none",
            ),
        ],
        className="w-fit h-fit flex flex-col gap-3 absolute left-[50%] top-[50%] -translate-x-[50%] -translate-y-[50%] bg-[#D2E9E9] z-[20]",
        style={"boxShadow": "0 0 30px 0px rgba(0, 0, 0, 0.50)", "display": "block"},
    )


@callback(
    Output("dialog_id", "style"),
    Input("close_btn", "n_clicks"),
    prevent_initial_call=True,
)
def closeDialog(n_clicks):
    return {"display": "none"}


@callback(
    Output("form-output", "children"),
    Input("submit", "n_clicks"),
    [
        State("user-form-name", "value"),
        State("user-form-email", "value"),
        State("user-form-age", "value"),
    ],
    prevent_initial_call=True,
)
def process_form(n_clicks, name, email, age):
    return "Form submitted successfully! Name: {name}, Email: {email}, Age: {age}"
