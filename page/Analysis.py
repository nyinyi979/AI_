from dash import html
from page.analysis.Clustering import Clustering
from page.analysis.DescriptiveAnalaysis import DescriptiveAnalysis
from page.analysis.Classifier import Classifier


def Analysis():
    return html.Div(
        [DescriptiveAnalysis(), Clustering(), Classifier()],
        className="grid grid-cols-1 2xl:grid-cols-2 gap-6 mx-20 my-20",
    )
