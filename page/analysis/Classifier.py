from dash import dcc, html, callback, Output, Input
import pandas as pd
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
                marks={i: f"{i:.1f}" for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
            ),
            html.Label("Select Train Size:"),
            dcc.Slider(
                id="train-size-slider",
                min=0.1,
                max=0.9,
                step=0.1,
                value=0.7,
                marks={i: f"{i:.1f}" for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
            ),
            html.Div(id="train-size-display", className="mt-2"),
            html.Div(id="test-size-display", className="mt-2"),
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
            html.Div(id="accuracy-display", className="mt-4 text-lg font-bold"),  # Placeholder for accuracy
        ],
        className="mb-4",
    )


@callback(
    [
        Output("classification-report-body", "children"),
        Output("train-size-display", "children"),
        Output("test-size-display", "children"),
        Output("accuracy-display", "children"),  # Output for accuracy
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
    if not file or "content" not in file:
        return (
            [html.Tr([html.Td("No data provided or invalid file format.", colSpan=5, className="p-2 border")])],
            f"Train Size: {train_size:.2f}",
            f"Test Size: {test_size:.2f}",
            "Accuracy: N/A",
        )

    df = pd.DataFrame(file["content"])

    if not xColumns or not yColumns:
        return (
            [html.Tr([html.Td("Please select both features and target columns.", colSpan=5, className="p-2 border")])],
            f"Train Size: {train_size:.2f}",
            f"Test Size: {test_size:.2f}",
            "Accuracy: N/A",
        )

    x = df[xColumns].select_dtypes(include="number")
    y = df[yColumns]

    if test_size + train_size > 1:
        train_size = 1 - test_size

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, train_size=train_size, random_state=42
        )
    except ValueError as e:
        return (
            [html.Tr([html.Td(f"Error in train-test split: {e}", colSpan=5, className="p-2 border")])],
            f"Train Size: {train_size:.2f}",
            f"Test Size: {test_size:.2f}",
            "Accuracy: N/A",
        )

    model = AdaBoostClassifier(random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    rows = []
    for idx, (label, metrics) in enumerate(report.items()):
        if label == "accuracy":
            continue
        rows.append(
            html.Tr(
                [
                    html.Td(str(label), className="p-2 border"),
                    html.Td(f"{metrics['precision']:.2f}", className="p-2 border"),
                    html.Td(f"{metrics['recall']:.2f}", className="p-2 border"),
                    html.Td(f"{metrics['f1-score']:.2f}", className="p-2 border"),
                    html.Td(f"{metrics['support']:.0f}", className="p-2 border"),
                ]
            )
        )

    return (
        rows,
        f"Train Size: {train_size:.2f}",
        f"Test Size: {test_size:.2f}",
        f"Accuracy: {accuracy:.2f}",  # Display accuracy under the table
    )


@callback(
    [Output("x-columns", "options"), Output("x-columns", "value"), Output("y-columns", "options"), Output("y-columns", "value")],
    Input("file-store", "data"),
)
def initialize_dropdowns(file):
    if file is None:
        return [], [], [], None

    df = pd.DataFrame(file["content"])
    if df.empty:
        return [], [], [], None

    # Get column options
    all_columns = df.columns
    numeric_columns = df.select_dtypes(include="number").columns

    # x-columns options and defaults
    x_options = [{"label": col, "value": col} for col in all_columns]
    x_default = list(numeric_columns)

    # y-columns options and default (first column)
    y_options = [{"label": col, "value": col} for col in all_columns]
    y_default = all_columns[0]

    return x_options, x_default, y_options, y_default
