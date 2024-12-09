from dash import dcc, html, callback, Output, Input
import pandas as pd
from page.analysis.ClassifierDialog import ClassifierDialog
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from components.Typography import P
from components.Button import Button


def Classifier():
    return html.Div(
        [
            dcc.Store(id="file-store", storage_type="local"),
            ClassifierDialog(),
            P(
                children=[html.Div(id="classifier-title", className="my-4")],
                variant="body1",
            ),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Th("Values", className="p-2"),
                                        html.Th("Precision", className="p-2"),
                                        html.Th("Recall", className="p-2"),
                                        html.Th("F1-Score", className="p-2"),
                                        html.Th("Support", className="p-2"),
                                    ]
                                ),
                            ],
                            className="font-semibold sticky bg-[#eeffffa3] backdrop-blur-sm border-b border-gray-200 top-0 left-0",
                        ),
                        html.Tbody(id="classification-report-body"),
                    ],
                    className="w-full border-collapse",
                ),
                className="max-h-[400px] overflow-y-auto relative",
            ),
            html.Div(
                id="accuracy-display", className="mt-4 text-lg font-bold"
            ),  # Placeholder for accuracy
            Button(
                children=[
                    "Setting",
                    html.Img(src="assets/images/setting.svg", className="size-6"),
                ],
                size="sm",
                variant="primary",
                className="w-fit flex gap-2 my-4",
                id="Classifier-setting",
                n_clicks=0,
            ),
        ],
        className="mb-4",
    )


@callback(
    [
        Output("classification-report-body", "children"),
        Output("accuracy-display", "children"),  # Output for accuracy
        Output("classifier-title", "children"),  # Output for classifier title
    ],
    [
        Input("file-store", "data"),
        Input("x-columns", "value"),
        Input("y-columns", "value"),
        Input("test-size-slider", "value"),
        Input("train-size-slider", "value"),
        Input("classifier-dropdown", "value"),  # Input for the selected classifier
    ],
)
def classifier(file, xColumns, yColumns, test_size, train_size, classifier_type):
    if not file or "content" not in file:
        return (
            [
                html.Tr(
                    [
                        html.Td(
                            "No data provided or invalid file format.",
                            colSpan=5,
                            className="p-2 border",
                        )
                    ]
                )
            ],
            f"Train Size: {train_size:.2f}",
            f"Test Size: {test_size:.2f}",
            "Accuracy: N/A",
            f"Report Using {classifier_type.capitalize()} Classifier",  # Classifier title
        )

    df = pd.DataFrame(file["content"])

    if not xColumns or not yColumns:
        return (
            [
                html.Tr(
                    [
                        html.Td(
                            "Please select both features and target columns.",
                            colSpan=5,
                            className="p-2 border",
                        )
                    ]
                )
            ],
            f"Train Size: {train_size:.2f}",
            f"Test Size: {test_size:.2f}",
            "Accuracy: N/A",
            f"Report Using {classifier_type.capitalize()} Classifier",  # Classifier title
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
            [
                html.Tr(
                    [
                        html.Td(
                            f"Error in train-test split: {e}",
                            colSpan=5,
                            className="p-2 border",
                        )
                    ]
                )
            ],
            f"Train Size: {train_size:.2f}",
            f"Test Size: {test_size:.2f}",
            "Accuracy: N/A",
            f"Report Using {classifier_type.capitalize()} Classifier",  # Classifier title
        )

    # Select model based on the classifier type
    if classifier_type == "adaboost":
        model = AdaBoostClassifier(random_state=42)
    elif classifier_type == "randomforest":
        model = RandomForestClassifier(random_state=42)

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
        f"Accuracy: {accuracy:.2f}",  # Display accuracy under the table
        f"Report Using {classifier_type.capitalize()} Classifier",  # Classifier title
    )


@callback(
    [
        Output("x-columns", "options"),
        Output("x-columns", "value"),
        Output("y-columns", "options"),
        Output("y-columns", "value"),
    ],
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


# Callback to adjust train and test size sliders based on the selected values
@callback(
    [
        Output("train-size-slider", "value"),
        Output("test-size-slider", "value"),
    ],
    [
        Input("train-size-slider", "value"),
        Input("test-size-slider", "value"),
    ],
)
def adjust_slider_values(train_size, test_size):
    if train_size + test_size > 1:
        # Adjust the other slider if the sum exceeds 1
        if train_size > test_size:
            # Lower the test size if train size is greater
            test_size = 1 - train_size
        else:
            # Lower the train size if test size is greater
            train_size = 1 - test_size

    return train_size, test_size
