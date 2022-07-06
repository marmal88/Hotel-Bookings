import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from src.app_func.preprocess import Preprocesor
from src.app_func.mlpipe import MLpipeline


# Data
df = Preprocesor().preprocess_df()
drop = [
    "booking_id",
    "booking_month",
    "arrival_month",
    "arrival_day",
    "checkout_month",
    "checkout_day",
    "price",
    "num_adults",
    "num_children",
    "currency",
    "room",
]
df = df.drop(labels=drop, axis=1)
countries = df["country"].unique().tolist()
platforms = df["platform"].unique().tolist()
branches = df["branch"].unique().tolist()
first_times = df["first_time"].unique().astype(str).tolist()
mean_price = round(df["SGD_price"].mean(), 2)


# Instantiate Application
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SANDSTONE],
    suppress_callback_exceptions=True,
)

# Application Layout
navbar = dbc.Nav(
    [
        dbc.NavItem(
            dbc.NavLink("Candidate Information & Agenda", href="/", active="exact")
        ),
        dbc.NavItem(dbc.NavLink("Front-end Application", href="/app", active="exact")),
    ],
    pills=True,
    justified=True,
)

content = html.Div(id="page-content", children=[])

app.layout = html.Div([dcc.Location(id="url"), navbar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return [
            dbc.Col(
                [
                    html.H1("Candidate Information", style={"textAlign": "left"}),
                    html.H4("Name: Low Guangwen Daniel", style={"textAlign": "left"}),
                    html.H4("Email: dlow017@e.ntu.edu.sg", style={"textAlign": "left"}),
                ],
                width={"size": 9, "offset": 2},
            ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1("Interview Agenda", style={"textAlign": "Left"}),
                            dcc.Markdown(
                                """
                                ##### 1) Application Demonstration
                                ##### 2) Review of Thought Process and Work done
                                - Exploratory Data Analysis (EDA) 
                                ##### 3) Code Review
                                - End-to-end Machine Learning Pipeline (MLP)
                                ##### 4) Other items
                                - CI workflow
                                - Unit-testing
                                """
                            ),
                        ],
                        width={"size": 9, "offset": 2},
                    )
                ],
                className="pad-row",
            ),
        ]
    elif pathname == "/app":
        return [
            dcc.ConfirmDialog(
                id="confirm-no-blanks",
                message="Please ensure that an amount is provided",
            ),
            html.H2("Hotel Room Booking Prediction", style={"textAlign": "center"}),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3(
                                "Input Booking Details", style={"textAlign": "Center"}
                            ),
                            html.Hr(),
                            html.H4(
                                "Please select customer's country of origin",
                                style={"textAlign": "Left", "margin-top": "25px"},
                            ),
                            dcc.Dropdown(
                                sorted(countries), "China", id="country-dropdown"
                            ),
                            html.H4(
                                "Please the platform the booking was made on",
                                style={"textAlign": "Left", "margin-top": "25px"},
                            ),
                            dcc.Dropdown(
                                sorted(platforms), "Website", id="platform-dropdown"
                            ),
                            html.H4(
                                "Please input total amount paid for stay (SGD)",
                                style={"textAlign": "Left", "margin-top": "25px"},
                            ),
                            dcc.Input(
                                id="amount-input",
                                type="text",
                                placeholder=mean_price,
                                debounce=True,
                            ),
                            html.H4(
                                "Please select location of booking",
                                style={"textAlign": "Left", "margin-top": "25px"},
                            ),
                            dcc.RadioItems(
                                branches,
                                "Changi",
                                id="branch-radio",
                                labelStyle={"display": "block"},
                            ),
                            html.H4(
                                "Please select if this is a first time booking",
                                style={"textAlign": "Left", "margin-top": "25px"},
                            ),
                            dcc.RadioItems(
                                first_times,
                                "True",
                                id="first_time-radio",
                                labelStyle={"display": "block"},
                            ),
                        ],
                        width={"size": 4, "offset": 1},
                    ),
                    dbc.Col(
                        [
                            html.H3("Prediction", style={"textAlign": "Center"}),
                            html.Hr(),
                            html.H4(
                                "Please click submit to make a prediction",
                                style={"textAlign": "Left", "margin-top": "25px"},
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "Submit",
                                        id="submit-button",
                                        n_clicks=0,
                                        color="success",
                                        className="me-1",
                                        style={"textAlign": "center"},
                                    )
                                ],
                                className="d-grid gap-2 col-2 mx-auto",
                            ),
                            dbc.Card(
                                [
                                    html.H4(
                                        "Results",
                                        style={
                                            "textAlign": "Left",
                                            "margin-top": "25px",
                                        },
                                    ),
                                    html.Hr(),
                                    html.P(
                                        id="model-name",
                                        style={
                                            "textAlign": "Left",
                                            "margin-top": "25px",
                                        },
                                    ),
                                    html.P(
                                        id="prediction",
                                        style={
                                            "textAlign": "Left",
                                            "margin-top": "25px",
                                        },
                                    ),
                                    html.P(
                                        id="predict-proba",
                                        style={
                                            "textAlign": "Left",
                                            "margin-top": "25px",
                                        },
                                    ),
                                ],
                                style={"textAlign": "Left", "margin-top": "25px"},
                            ),
                        ],
                        width={"size": 4, "offset": 1},
                    ),
                ]
            ),
        ]
    # If the user tries to reach a different page, return a 404 message
    return [
        html.H1("404: Not found", className="text-danger"),
        html.Hr(),
        html.P(f"The pathname {pathname} was not recognised"),
    ]


@app.callback(
    Output("model-name", "children"),
    Output("prediction", "children"),
    Output("predict-proba", "children"),
    Input("submit-button", "n_clicks"),
    [
        State("country-dropdown", "value"),
        State("platform-dropdown", "value"),
        State("amount-input", "value"),
        State("branch-radio", "value"),
        State("first_time-radio", "value"),
    ],
)
def update_output(n_clicks, country, platform, amount, branch, first_time):
    if n_clicks > 0 and amount is not None:
        pred_row = pd.DataFrame(
            {
                "country": str(country),
                "platform": str(platform),
                "branch": str(branch),
                "first_time": bool(first_time),
                "SGD_price": float(amount),
            },
            index=[0],
        )
        model_name, y_pred_prob, y_pred_test = MLpipeline().frontend_output(
            df, pred_row
        )
        return (
            f"The model predicts No-Show to be {y_pred_test}.",
            f"The prediction probability is {y_pred_prob:.4f}.",
            f"The model used for prediction was {model_name}",
        )
    elif n_clicks == 0:
        raise PreventUpdate


@app.callback(
    Output("confirm-no-blanks", "displayed"),
    Input("submit-button", "n_clicks"),
    State("amount-input", "value"),
)
def amount_warning(n_clicks, amount):
    if amount is None and n_clicks > 0:
        if amount is None:
            n_clicks = 0
            return True
        return False


if __name__ == "__main__":
    app.run_server(debug=True)
