from dash import html
from components.Typography import P


def Classifier():
    return html.Div(
        [
            P(
                "Report Using AdaboostClassifier",
                variant="body1",
                className="mb-2",
            ),
            html.Table(
                [
                    html.Thead(
                        [
                            html.Tr(
                                [
                                    html.Th("No", className="p-2 border"),
                                    html.Th(
                                        "Precision",
                                        className="p-2 border",
                                    ),
                                    html.Th("Recall", className="p-2 border"),
                                    html.Th(
                                        "F1-Score",
                                        className="p-2 border",
                                    ),
                                    html.Th(
                                        "Support",
                                        className="p-2 border",
                                    ),
                                ]
                            ),
                        ]
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td("1", className="p-2 border"),
                                    html.Td("0.89", className="p-2 border"),
                                    html.Td("0.90", className="p-2 border"),
                                    html.Td("0.89", className="p-2 border"),
                                    html.Td("100", className="p-2 border"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("2", className="p-2 border"),
                                    html.Td("0.85", className="p-2 border"),
                                    html.Td("0.80", className="p-2 border"),
                                    html.Td("0.82", className="p-2 border"),
                                    html.Td("120", className="p-2 border"),
                                ]
                            ),
                        ]
                    ),
                    # Sample rows for precision, recall, etc.
                ],
                className="w-full border-collapse",
            ),
        ],
        className="mb-4",
    )
