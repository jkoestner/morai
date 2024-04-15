"""Experience dashboard."""

import json

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
import joblib
import pandas as pd
from dash_extensions.enrich import (
    ALL,
    Input,
    Output,
    Serverside,
    State,
    callback,
    callback_context,
    dcc,
    html,
)

from morai.dashboard.components import dash_formats
from morai.dashboard.utils import dashboard_helper as dh
from morai.experience import charters
from morai.forecast import metrics, preprocessors
from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)


dash.register_page(__name__, path="/model", title="morai - Model", order=3)


#   _                            _
#  | |    __ _ _   _  ___  _   _| |_
#  | |   / _` | | | |/ _ \| | | | __|
#  | |__| (_| | |_| | (_) | |_| | |_
#  |_____\__,_|\__, |\___/ \__,_|\__|
#              |___/


def layout():
    """Model layout."""
    return html.Div(
        [
            dcc.Store(id="store-model-results", storage_type="session"),
            # -----------------------------------
            html.H4(
                "Model Analysis",
                className="bg-primary text-white p-2 mb-2 text-center",
            ),
            dbc.Row(
                html.Div(id="data-description"),
            ),
            dbc.Toast(
                (
                    "Need to put results in the model_results.json "
                    "file to display results."
                ),
                id="toast-no-model-results",
                header="Input Error",
                is_open=False,
                dismissable=True,
                icon="danger",
                style={"position": "fixed", "top": 100, "right": 10, "width": 350},
            ),
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            html.Div(
                                id="model-results",
                            ),
                        ],
                        title="Model Results",
                    ),
                    dbc.AccordionItem(
                        [
                            html.Div(
                                id="model-scorecard",
                            ),
                        ],
                        title="Model Scorecard",
                    ),
                    dbc.AccordionItem(
                        [
                            html.Div(
                                id="model-importance",
                            ),
                        ],
                        title="Model Importance",
                    ),
                    dbc.AccordionItem(
                        [
                            html.Div(
                                id="model-target",
                            ),
                        ],
                        title="Model Target",
                    ),
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Create PDP",
                                                    id="button-pdp",
                                                    className="btn btn-primary p-1",
                                                ),
                                                html.H5(
                                                    "Selectors",
                                                    style={
                                                        "border-bottom": "1px solid black",
                                                        "padding-bottom": "5px",
                                                    },
                                                ),
                                                html.Div(
                                                    id="pdp-selectors",
                                                ),
                                            ],
                                            className="mt-2 bg-light border p-1",
                                        ),
                                        width=2,
                                    ),
                                    dbc.Col(
                                        dcc.Loading(
                                            id="loading-pdp-chart",
                                            type="dot",
                                            children=html.Div(id="pdp-chart"),
                                        ),
                                        width=8,
                                    ),
                                    dbc.Col(
                                        [
                                            html.H5(
                                                "Filters",
                                                style={
                                                    "border-bottom": "1px solid black",
                                                    "padding-bottom": "5px",
                                                },
                                            ),
                                            html.Div(
                                                id="pdp-filters",
                                            ),
                                        ],
                                        width=2,
                                        className="mt-2 bg-light border p-1",
                                    ),
                                ],
                            ),
                        ],
                        title="Model PDP",
                    ),
                ],
                start_collapsed=True,
                always_open=True,
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
        Output("store-model-results", "data"),
        Output("toast-no-model-results", "is_open"),
    ],
    Input("url", "pathname"),
)
def load_model_results(pathname):
    """Load model results."""
    if pathname != "/model":
        raise dash.exceptions.PreventUpdate

    if not (helpers.FILES_PATH / "result" / "model_results.json").exists():
        return dash.no_update, True

    # load models
    model_results = metrics.ModelResults(filepath="model_results.json")

    return Serverside(model_results, key="model_results"), False


@callback(
    [Output("pdp-selectors", "children"), Output("pdp-filters", "children")],
    Input("url", "pathname"),
    [State("store-config", "data"), State("store-dataset", "data")],
)
def load_pdp_selectors(pathname, config, dataset):
    """Load pdp selectors."""
    if pathname != "/model" or config is None:
        raise dash.exceptions.PreventUpdate

    # create the pdp selectors
    pdp_selectors = dh.generate_selectors(
        config=config,
        prefix="pdp",
        selector_dict={
            "model_file": True,
            "x_axis": True,
            "color": True,
            "secondary": True,
            "x_bins": True,
            "pdp_weight": True,
        },
    )

    pdp_filters = dh.generate_filters(df=dataset, prefix="pdp")["filters"]

    return pdp_selectors, pdp_filters


@callback(
    [
        Output("model-results", "children"),
        Output("model-scorecard", "children"),
        Output("model-importance", "children"),
        Output("model-target", "children"),
    ],
    [Input("url", "pathname"), Input("store-model-results", "data")],
)
def display_model_results(pathname, model_results):
    """Load model results."""
    if pathname != "/model" or model_results is None:
        raise dash.exceptions.PreventUpdate

    logger.debug("Display model results")
    # load models
    models = model_results.model
    scorecard = model_results.scorecard
    importance = model_results.importance
    target = model_results.importance[["model_name"]]

    # update column defs
    model_column_defs = dash_formats.get_column_defs(models)
    model_col_def = {
        "headerName": "model_name",
        "field": "model_name",
        "pinned": "left",
    }
    model_column_defs = dash_formats.remove_column_defs(model_column_defs, "model_name")
    model_column_defs.insert(0, model_col_def)

    score_column_defs = dash_formats.get_column_defs(scorecard)
    score_col_def = {
        "headerName": "model_name",
        "field": "model_name",
        "pinned": "left",
    }
    score_column_defs = dash_formats.remove_column_defs(score_column_defs, "model_name")
    score_column_defs.insert(0, score_col_def)

    importance_column_defs = dash_formats.get_column_defs(importance)

    target_column_defs = dash_formats.get_column_defs(target)

    # create the divs
    model_results_div = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dag.AgGrid(
                                id="grid-model-dictionary",
                                rowData=models.to_dict("records"),
                                columnDefs=model_column_defs,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        dcc.Markdown(
                            id="markdown-model-dictionary",
                            children=(
                                "### Model Details ",
                                "Select a cell to see value",
                            ),
                        ),
                        width=6,
                        style={"height": "400px", "overflowY": "auto"},
                    ),
                ],
            ),
        ],
    )
    model_scorecard_div = html.Div(
        [
            dag.AgGrid(
                rowData=scorecard.to_dict("records"),
                columnDefs=score_column_defs,
            ),
        ],
    )
    model_importance_div = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dag.AgGrid(
                                id="grid-importance-dictionary",
                                rowData=importance.to_dict("records"),
                                columnDefs=importance_column_defs,
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        dcc.Loading(
                            id="loading-importance-chart",
                            type="dot",
                            children=html.Div(id="importance-chart"),
                        ),
                        width=8,
                        style={"height": "400px", "overflowY": "auto"},
                    ),
                ],
            ),
        ],
    )
    model_target_div = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dag.AgGrid(
                                id="grid-target-dictionary",
                                rowData=target.to_dict("records"),
                                columnDefs=target_column_defs,
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        dcc.Loading(
                            id="loading-target-chart",
                            type="dot",
                            children=html.Div(id="target-chart"),
                        ),
                        width=10,
                        style={"height": "400px", "overflowY": "auto"},
                    ),
                ],
            ),
        ],
    )
    return (
        model_results_div,
        model_scorecard_div,
        model_importance_div,
        model_target_div,
    )


@callback(
    Output("markdown-model-dictionary", "children"),
    Input("grid-model-dictionary", "cellClicked"),
)
def clicked_cell_model_dictionary(cell):
    """Clicked cell model dictionary."""
    if not cell:
        raise dash.exceptions.PreventUpdate

    markdown = json.dumps(cell["value"], indent=2)
    markdown = f"```json\n{markdown}\n```"
    return markdown


@callback(
    Output("importance-chart", "children"),
    Input("grid-importance-dictionary", "cellClicked"),
)
def clicked_cell_importance_dictionary(cell):
    """Clicked cell importance dictionary."""
    if not cell or cell["colId"] != "importance":
        raise dash.exceptions.PreventUpdate

    if cell["value"] is None:
        return dcc.Markdown("No data to display")

    columns = cell["value"]["columns"]
    data = cell["value"]["data"]
    importance = pd.DataFrame(data, columns=columns)

    importance_chart = charters.chart(
        df=importance, x_axis="feature", y_axis="importance", type="bar", y_sort=True
    )

    return dcc.Graph(figure=importance_chart)


@callback(
    Output("target-chart", "children"),
    Input("grid-target-dictionary", "cellClicked"),
    [State("store-dataset", "data"), State("store-config", "data")],
)
def clicked_cell_target_dictionary(cell, model_data, config):
    """Clicked cell target dictionary."""
    if not cell:
        raise dash.exceptions.PreventUpdate

    if cell["value"] is None:
        return dcc.Markdown("No data to display")

    # target chart parameters
    model = cell["value"]
    config_dataset = config["datasets"][config["general"]["dataset"]]
    features = config_dataset["columns"]["features"]
    numerator = config_dataset["columns"]["actuals_amt"]
    denmoninator = f"exp_amt_{model}"

    # create target chart
    target_chart = charters.target(
        df=model_data,
        target="ratio",
        cols=3,
        features=features,
        numerator=numerator,
        denominator=denmoninator,
    )

    return dcc.Graph(figure=target_chart)


@callback(
    Output("pdp-chart", "children"),
    Input("button-pdp", "n_clicks"),
    [
        State("store-model-results", "data"),
        State("store-dataset", "data"),
        State({"type": "pdp-selector", "index": ALL}, "value"),
        State({"type": "pdp-str-filter", "index": ALL}, "value"),
        State({"type": "pdp-num-filter", "index": ALL}, "value"),
    ],
)
def display_pdp(
    n_clicks, model_results, model_data, pdp_selectors, pdp_str_filters, pdp_num_filters
):
    """Create pdp."""
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # get the callback context
    states_info = dh._inputs_flatten_list(callback_context.states_list)
    model_file = dh._inputs_parse_id(states_info, "model_file_selector")
    x_axis_col = dh._inputs_parse_id(states_info, "x_axis_selector")

    # checks inputs needed
    if model_file is None or x_axis_col is None:
        return dash.no_update

    # load model
    model = joblib.load(helpers.FILES_PATH / "models" / model_file)
    model_name = model_file.split(".")[0]

    # get model parameters
    standardize = model_results.model[
        model_results.model["model_name"] == model_name
    ].iloc[0]["preprocess_params"]["standardize"]
    preset = model_results.model[model_results.model["model_name"] == model_name].iloc[
        0
    ]["preprocess_params"]["preset"]
    feature_dict = model_results.model[
        model_results.model["model_name"] == model_name
    ].iloc[0]["feature_dict"]

    # filter_data
    filtered_df = dh.filter_data(df=model_data, callback_context=states_info)

    # encode data
    preprocess_dict = preprocessors.preprocess_data(
        filtered_df,
        feature_dict=feature_dict,
        standardize=standardize,
        preset=preset,
    )
    mapping = preprocess_dict["mapping"]
    md_encoded = preprocess_dict["md_encoded"]

    # create pdp
    pdp_chart = charters.pdp(
        model=model,
        df=md_encoded,
        x_axis=x_axis_col,
        line_color=dh._inputs_parse_id(states_info, "color_selector"),
        weight=dh._inputs_parse_id(states_info, "weights_selector"),
        secondary=dh._inputs_parse_id(states_info, "secondary_selector"),
        mapping=mapping,
        x_bins=dh._inputs_parse_id(states_info, "x_bins_selector"),
    )

    return dcc.Graph(figure=pdp_chart)
