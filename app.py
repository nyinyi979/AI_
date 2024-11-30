import dash
from dash import Dash, html
from components.Button import Button

app = Dash(__name__, use_pages=True)

app.layout = html.Div(
    [
        html.Div(
            [
                Button(
                    children="Home Page",
                    variant="primary_ghost",
                    size="sm",
                    asLink=True,
                    href="/",
                ),
                Button(
                    children="Analysis Page",
                    variant="primary_ghost",
                    size="sm",
                    asLink=True,
                    href="/analysis",
                ),
                Button(
                    children=["Login", html.Img(src="/assets/images/login.svg")],
                    variant="primary",
                    size="sm",
                    asLink=True,
                    href="/login",
                    className="flex items-center gap-2.5",
                ),
                Button(
                    children="Signup",
                    variant="primary_ghost",
                    size="sm",
                    asLink=True,
                    href="/signup",
                ),
            ],
            style={"boxShadow": "0 0 25px 0 hsla(180, 30%, 82%, 0.50)"},
            className="w-full sticky top-0 left-0 flex gap-[30px] justify-end items-center px-20 py-5 backdrop-blur-sm bg-[#EEFFFF]/80 z-[100]",
        ),
        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
