import pandas as pd
import numpy as np
import plotly.graph_objs as go

from page.analysis.ClusteringDialog import ClusteringDialog
from components.Button import Button
from dash import html, dcc, Input, Output, callback
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from components.Typography import P


def Clustering():
    return html.Div(
        [
            dcc.Store(id="file-store", storage_type="local"),
            ClusteringDialog(),
            P(
                "K-Means Clustering",
                variant="body1",
                className="mb-4",
            ),
            html.Div(
                [
                    # Right Column: Visualizations
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="unlabeled-data",
                                        figure={
                                            "layout": go.Layout(
                                                title="Original Data",
                                                plot_bgcolor="rgba(0,0,0,0)",
                                                paper_bgcolor="rgba(0,0,0,0)",
                                            )
                                        },
                                        config={
                                            "displayModeBar": False,
                                            "displaylogo": False,
                                        },
                                    ),
                                    P(
                                        "Unlabeled Data",
                                        variant="body1",
                                        className="text-center",
                                    ),
                                ],
                            ),
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="clustered-data",
                                        config={
                                            "displayModeBar": False,
                                            "displaylogo": False,
                                        },
                                    ),
                                    P(
                                        "Clustered Data",
                                        variant="body1",
                                        className="text-center",
                                    ),
                                ],
                            ),
                        ],
                        className="w-full grid grid-cols-2",
                    ),
                ],
            ),
            Button(
                children=[
                    "Setting",
                    html.Img(src="assets/images/setting.svg", className="size-6"),
                ],
                size="sm",
                variant="primary",
                className="w-fit flex gap-2",
                id="clustering-setting",
                n_clicks=0
            ),
        ],
        className="flex flex-col gap-2 px-4 pt-4 pb-8 relative border-b border-[#B1CBCB] 2xl:border-none",
    )


@callback(
    [
        Output("unlabeled-data", "figure"),
        Output("clustered-data", "figure"),
        Output("cluster-summary", "children"),
    ],
    [
        Input("n-clusters-slider", "value"),
        Input("scaling-method-dropdown", "value"),
        Input("file-store", "data"),
    ],
)
def update_clustering_visualization(n_clusters, scaling_method, file_data):
    """
    Callback to update clustering visualizations based on user inputs
    """
    if not file_data:
        # Return default figures if no data uploaded
        default_layout = go.Layout(
            title="Please Upload Data",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return (
            go.Figure(layout=default_layout),
            go.Figure(layout=default_layout),
            P("No data uploaded", variant="body2"),
        )

    try:
        df = pd.DataFrame(file_data["content"])

        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numerical_cols]

        # Standardize/Scale data
        if scaling_method == "standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Unlabeled data scatter plot
        unlabeled_fig = go.Figure(
            data=go.Scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                mode="markers",
                marker=dict(color="blue", size=8),
                text=df.index,
                hoverinfo="text",
            ),
            layout=go.Layout(
                title="Original Data (PCA Reduced)",
                xaxis_title="First Principal Component",
                yaxis_title="Second Principal Component",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            ),
        )

        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Clustered data scatter plot
        clustered_fig = go.Figure(
            data=go.Scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                mode="markers",
                marker=dict(color=cluster_labels, colorscale="Viridis", size=8),
                text=[f"Cluster: {label}" for label in cluster_labels],
                hoverinfo="text",
            ),
            layout=go.Layout(
                title=f"K-Means Clustering (k={n_clusters})",
                xaxis_title="First Principal Component",
                yaxis_title="Second Principal Component",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            ),
        )

        # Cluster Summary
        cluster_summary_df = pd.DataFrame(
            {
                "Cluster": range(n_clusters),
                "Size": [sum(cluster_labels == i) for i in range(n_clusters)],
            }
        )

        cluster_summary_children = [
            P("Cluster Summary", variant="body1", className="mb-2"),
            html.Table(
                children=[
                    html.Thead(
                        children=html.Tr(
                            [html.Th(col) for col in cluster_summary_df.columns]
                        )
                    ),
                    html.Tbody(
                        children=[
                            html.Tr(
                                [
                                    html.Td(cluster_summary_df.iloc[i][col])
                                    for col in cluster_summary_df.columns
                                ]
                            )
                            for i in range(len(cluster_summary_df))
                        ]
                    ),
                ]
            ),
        ]

        return unlabeled_fig, clustered_fig, cluster_summary_children

    except Exception as e:
        # Error handling
        error_layout = go.Layout(
            title=f"Error: {str(e)}",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return (
            go.Figure(layout=error_layout),
            go.Figure(layout=error_layout),
            P(
                f"Error processing data: {str(e)}",
                variant="body2",
                className="text-red-500",
            ),
        )

@callback(
    Output("clustering-dialog","style"),
    Input("clustering-setting","n_clicks"),
    prevent_initial_callback=True
)
def openClusteringDialog(n_clicks):
    if n_clicks:
        return {"boxShadow": "0 0 30px 0px rgba(0, 0, 0, 0.50)", "display": "block"}
    else:
        return None
