"""Explore dashboard."""

import dash_extensions.enrich as dash
from dash_extensions.enrich import (
    Input,
    Output,
    State,
    callback,
    dcc,
    html,
)

from morai.dashboard.components import sidebars
from morai.experience import charters
from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)

dash.register_page(__name__, path="/explore", title="morai - Explore", order=1)

#   _                            _
#  | |    __ _ _   _  ___  _   _| |_
#  | |   / _` | | | |/ _ \| | | | __|
#  | |__| (_| | |_| | (_) | |_| | |_
#  |_____\__,_|\__, |\___/ \__,_|\__|
#              |___/


def layout():
    """Explore layout."""
    return html.Div(
        [
            html.Div(  # Sidebar container
                sidebars.explore_sidebar,
            ),
            html.Div(  # Main content container
                [
                    html.H4(
                        "Explore Data",
                        className="bg-primary text-white p-2 mb-2 text-center",
                    ),
                    html.Div(
                        [
                            html.Div(
                                id="data-description",
                            ),
                            html.H5(
                                "Frequency Number",
                                id="section-freq-num",
                                className="bg-secondary text-white p-2 mb-2",
                            ),
                            dcc.Loading(
                                id="loading-chart-freq-num",
                                type="dot",
                                children=html.Div(id="chart-freq-num"),
                            ),
                            html.H5(
                                "Frequency Categorical",
                                id="section-freq-cat",
                                className="bg-secondary text-white p-2 mb-2",
                            ),
                            dcc.Loading(
                                id="loading-chart-freq-cat",
                                type="dot",
                                children=html.Div(id="chart-freq-cat"),
                            ),
                            html.H5(
                                "Target",
                                id="section-target",
                                className="bg-secondary text-white p-2 mb-2",
                            ),
                            dcc.Loading(
                                id="loading-chart-target",
                                type="dot",
                                children=html.Div(id="chart-target"),
                            ),
                        ],
                    ),
                ],
                style={"marginLeft": "250px"},
            ),
        ],
        className="container",
    )


#    ____      _ _ _                _
#   / ___|__ _| | | |__   __ _  ___| | _____
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/


@callback(
    [
        Output("nav-freq-num", "active"),
        Output("nav-freq-cat", "active"),
        Output("nav-target", "active"),
    ],
    [Input("url", "hash"), Input("url", "href")],
)
def set_active_link(hash, href):
    """Set the active link in the sidebar based on the URL."""
    if not hash:
        return True, False, False
    elif "#section-freq-num" in hash:
        return True, False, False
    elif "#section-freq-cat" in hash:
        return False, True, False
    elif "#section-target" in hash:
        return False, False, True
    return False, False, False


@callback(
    [
        Output("chart-freq-num", "children"),
        Output("chart-freq-cat", "children"),
        Output("chart-target", "children"),
    ],
    [Input("url", "pathname")],
    [State("store-dataset", "data"), State("store-config", "data")],
)
def update_charts(pathname, dataset, config):
    """Update charts when page is loaded."""
    if pathname != "/explore" or dataset is None:
        raise dash.exceptions.PreventUpdate
    logger.debug("update explore charts")
    config_dataset = config["datasets"][config["general"]["dataset"]]
    features = config_dataset["columns"]["features"]
    num_cols = dataset[features].select_dtypes(include=["number"]).columns.tolist()
    # charts
    chart_freq_num = charters.frequency(
        dataset,
        cols=3,
        features=num_cols,
        sum_var=config_dataset["columns"]["exposure_amt"],
    )

    chart_freq_cat = charters.frequency(dataset, cols=3, sum_var="amount_exposed")

    chart_target = charters.target(
        df=dataset,
        target="risk",
        cols=3,
        features=features,
        numerator=[config_dataset["defaults"]["numerator"]],
        denominator=[config_dataset["defaults"]["denominator"]],
    )

    chart_freq_num = dcc.Graph(figure=chart_freq_num)
    chart_freq_cat = dcc.Graph(figure=chart_freq_cat)
    chart_target = dcc.Graph(figure=chart_target)

    return chart_freq_num, chart_freq_cat, chart_target


@callback(
    [
        Output("data-description", "children"),
    ],
    [Input("url", "pathname")],
    [State("store-config", "data")],
)
def update_dataset_description(pathname, config):
    """Update charts when page is loaded."""
    if pathname not in ["/explore", "/experience", "/model"]:
        raise dash.exceptions.PreventUpdate

    filename = "No dataset loaded"
    description = "No dataset loaded"
    if config is not None:
        config_dataset = config["datasets"][config["general"]["dataset"]]
        filename = f"{config_dataset['filename']}"
        description = f"{config_dataset['description']}"

    div_description = html.Div(
        html.P(
            [
                html.Span(
                    "Dataset: ",
                    style={"fontWeight": "bold"},
                ),
                filename,
                html.Br(),
                html.Span(
                    "Description: ",
                    style={"fontWeight": "bold"},
                ),
                description,
            ],
        ),
    )

    return div_description
