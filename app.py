import dash
from dash import Dash, html, dcc

app = Dash(__name__, use_pages=True)

app.layout = html.Div(
    [
        html.Div(
            [
                html.A(
                    "Home Page",
                    href="/",
                    className="p-2.5 rounded-[10px] font-semibold border border-[#B1CBCB] hover:bg-[#E3F4F4] duration-300",
                ),
                html.A(
                    "Analysis Page",
                    href="/analysis",
                    className="p-2.5 rounded-[10px] font-semibold border border-[#B1CBCB] hover:bg-[#E3F4F4] duration-300",
                ),
                html.Button(
                    children=["Login", html.Img(src="/assets/images/login.svg")],
                    className="flex items-center gap-2.5 p-2.5 rounded-[10px] font-semibold bg-[#C4DFDF] hover:bg-[#B1CBCB] duration-300",
                ),
                html.Button(
                    "Signup",
                    className="p-2.5 rounded-[10px] font-semibold border border-[#B1CBCB] hover:bg-[#E3F4F4] duration-300",
                ),
            ],
            style={"box-shadow": "0 0 25px 0 hsla(180, 30%, 82%, 0.50)"},
            className="w-full sticky top-0 left-0 flex gap-[30px] relative justify-end items-center px-20 py-5 opacity-[80%] bg-[#EEFFFF] z-[100]",
        ),
        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
