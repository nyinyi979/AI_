import pandas as pd

from dash import dcc, html, callback, Output, Input, State
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def Classifier():
    return html.Div(
        [
            dcc.Store(id="file-store", storage_type="local"),
            html.P("Report Using AdaboostClassifier", className="mb-2"),
            html.Label("Select Features (x):"),
            dcc.Dropdown(
                id="x-columns", multi=True, placeholder="Select feature columns"
            ),
            html.Label("Select Target (y):"),
            dcc.Dropdown(
                id="y-columns", multi=False, placeholder="Select target column"
            ),
            html.Label("Select Test Size:"),
            dcc.Slider(
                id="test-size-slider",
                min=0.1,
                max=0.9,
                step=0.1,
                value=0.3,  # Default value
                marks={
                    i: f"{i:.1f}" for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                },
            ),
            html.Label("Select Train Size:"),
            dcc.Slider(
                id="train-size-slider",
                min=0.1,
                max=0.9,
                step=0.1,
                value=0.7,  # Default value
                marks={
                    i: f"{i:.1f}" for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                },
            ),
            html.Table(
                [
                    html.Thead(
                        [
                            html.Tr(
                                [
                                    html.Th("No", className="p-2 border"),
                                    html.Th("Precision", className="p-2 border"),
                                    html.Th("Recall", className="p-2 border"),
                                    html.Th("F1-Score", className="p-2 border"),
                                    html.Th("Support", className="p-2 border"),
                                ]
                            ),
                        ]
                    ),
                    html.Tbody(id="classification-report-body"),
                ],
                className="w-full border-collapse",
            ),
            # Add Divs for displaying the sizes
            html.Div(id="train-size-display", className="mt-2"),
            html.Div(id="test-size-display", className="mt-2"),
        ],
        className="mb-4",
    )


@callback(
    [
        Output("classification-report-body", "children"),
        Output("train-size-display", "children"),
        Output("test-size-display", "children"),
    ],
    [
        Input("file-store", "data"),
        Input("x-columns", "value"),
        Input("y-columns", "value"),
        Input("test-size-slider", "value"),
        Input("train-size-slider", "value"),
    ],
)
def adaboostClassifier(file, xColumns, yColumns, test_size, train_size):
    if file is None:
        return (
            html.Tr([html.Td("No file uploaded", colSpan=5, className="p-2 border")]),
            f"Train Size: N/A",
            f"Test Size: N/A",
        )

    df = pd.DataFrame(file["content"])

    # Validate x_columns and y_column
    if xColumns is None or yColumns is None:
        return (
            html.Tr(
                [
                    html.Td(
                        "Please select features and target",
                        colSpan=5,
                        className="p-2 border",
                    )
                ]
            ),
            f"Train Size: N/A",
            f"Test Size: N/A",
        )

    if not set(xColumns).issubset(df.columns) or yColumns not in df.columns:
        return (
            html.Tr(
                [html.Td("Invalid column selection", colSpan=5, className="p-2 border")]
            ),
            f"Train Size: N/A",
            f"Test Size: N/A",
        )

    # Ensure test_size + train_size <= 1
    if test_size + train_size > 1:
        train_size = 1 - test_size

    # Prepare data
    x = df[xColumns]
    y = df[yColumns]

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, train_size=train_size, random_state=42
        )
    except ValueError as e:
        return (
            html.Tr(
                [
                    html.Td(
                        f"Error in train-test split: {e}",
                        colSpan=5,
                        className="p-2 border",
                    )
                ]
            ),
            f"Train Size: {train_size:.2f}",
            f"Test Size: {test_size:.2f}",
        )

    # Train AdaBoost Classifier
    model = AdaBoostClassifier(random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Generate table rows dynamically
    rows = []
    for idx, (label, metrics) in enumerate(report.items()):
        if label == "accuracy":
            continue
        rows.append(
            html.Tr(
                [
                    html.Td(
                        label if label.isdigit() else "Average", className="p-2 border"
                    ),
                    html.Td(f"{metrics['precision']:.2f}", className="p-2 border"),
                    html.Td(f"{metrics['recall']:.2f}", className="p-2 border"),
                    html.Td(f"{metrics['f1-score']:.2f}", className="p-2 border"),
                    html.Td(f"{metrics['support']:.0f}", className="p-2 border"),
                ]
            )
        )
    return rows, f"Train Size: {train_size:.2f}", f"Test Size: {test_size:.2f}"


@callback(
    Output("x-columns", "options"),
    Output("y-columns", "options"),
    Input("file-store", "data"),
)
def outputOption(file):
    if file is None:
        return [], []

    df = pd.DataFrame(file["content"])
    x_options = df.select_dtypes(include="number").dropna().columns.tolist()
    y_options = df.select_dtypes(include="number").dropna().columns.tolist()
    return x_options, y_options
