"""Models dashboard."""

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


def layout():
    """Model layout."""
    return html.Div(
        [
            dcc.Store(id="store-model-results", storage_type="session"),
            # Header section with gradient background
            html.Div(
                [
                    html.Div(
                        [
                            html.H4(
                                [
                                    html.I(className="fas fa-brain me-2"),
                                    "Model Analysis",
                                ],
                                className="mb-1",
                            ),
                            html.P(
                                "Analyze and evaluate model performance",
                                className="text-white-50 mb-0 small",
                            ),
                        ],
                        className="bg-gradient bg-primary text-white p-4 mb-4 rounded-3 shadow-sm",
                    ),
                ],
            ),
            # Dataset description
            dbc.Row(
                html.Div(
                    id="data-description",
                    className="mb-4",
                ),
            ),
            # Toast notifications
            dbc.Toast(
                id="models-toast",
                header="Notification",
                is_open=False,
                dismissable=True,
                icon="danger",
                style={"position": "fixed", "top": 100, "right": 10, "width": 350},
            ),
            # Main Content Accordion
            dbc.Accordion(
                [
                    # Model Results Section
                    dbc.AccordionItem(
                        dbc.Card(
                            dbc.CardBody(
                                html.Div(id="model-results"),
                            ),
                            className="shadow-sm border-0",
                        ),
                        title=[
                            html.I(className="fas fa-chart-line me-2"),
                            "Model Results",
                        ],
                    ),
                    # Model Scorecard Section
                    dbc.AccordionItem(
                        dbc.Card(
                            dbc.CardBody(
                                html.Div(id="model-scorecard"),
                            ),
                            className="shadow-sm border-0",
                        ),
                        title=[
                            html.I(className="fas fa-award me-2"),
                            "Model Scorecard",
                        ],
                    ),
                    # Model Importance Section
                    dbc.AccordionItem(
                        dbc.Card(
                            dbc.CardBody(
                                html.Div(id="model-importance"),
                            ),
                            className="shadow-sm border-0",
                        ),
                        title=[
                            html.I(className="fas fa-weight-hanging me-2"),
                            "Model Importance",
                        ],
                    ),
                    # Model Target Section
                    dbc.AccordionItem(
                        dbc.Card(
                            dbc.CardBody(
                                html.Div(id="model-target"),
                            ),
                            className="shadow-sm border-0",
                        ),
                        title=[
                            html.I(className="fas fa-bullseye me-2"),
                            "Model Target",
                        ],
                    ),
                    # PDP Analysis Section
                    dbc.AccordionItem(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            # Selectors Column
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader(
                                                            html.H5(
                                                                [
                                                                    html.I(
                                                                        className="fas fa-sliders-h me-2"
                                                                    ),
                                                                    "Analysis Options",
                                                                ],
                                                                className="mb-0",
                                                            ),
                                                            className="bg-light",
                                                        ),
                                                        dbc.CardBody(
                                                            [
                                                                dbc.Button(
                                                                    [
                                                                        html.I(
                                                                            className="fas fa-sync-alt me-2"
                                                                        ),
                                                                        "Create PDP",
                                                                    ],
                                                                    id="button-pdp",
                                                                    color="primary",
                                                                    className="w-100 shadow-sm mb-3",
                                                                ),
                                                                html.Div(
                                                                    id="pdp-selectors"
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    className="shadow-sm h-100",
                                                ),
                                                width=3,
                                            ),
                                            # Chart Column with Filter Button
                                            dbc.Col(
                                                [
                                                    # Filter Button
                                                    dbc.Button(
                                                        [
                                                            html.I(
                                                                className="fas fa-filter me-2"
                                                            ),
                                                            "Show Filters",
                                                        ],
                                                        id="open-pdp-filters-button",
                                                        className="mb-3",
                                                        color="primary",
                                                    ),
                                                    # Chart
                                                    dcc.Loading(
                                                        id="loading-pdp-chart",
                                                        type="default",
                                                        color="#007bff",
                                                        children=html.Div(
                                                            id="pdp-chart",
                                                            className="bg-white rounded-3 shadow-sm p-3",
                                                        ),
                                                    ),
                                                ],
                                                width=9,
                                            ),
                                        ],
                                        className="g-3",
                                    ),
                                ]
                            ),
                            className="shadow-sm border-0",
                        ),
                        title=[
                            html.I(className="fas fa-chart-area me-2"),
                            "Partial Dependence Plots",
                        ],
                    ),
                ],
                start_collapsed=True,
                always_open=True,
                className="shadow-sm",
            ),
            # Add Filters Offcanvas
            dbc.Offcanvas(
                [
                    html.H4(
                        [
                            html.I(className="fas fa-filter me-2"),
                            "Data Filters",
                        ],
                        className="mb-4",
                    ),
                    html.Button(
                        [
                            html.I(className="fas fa-undo me-2"),
                            "Reset Filters",
                        ],
                        id="reset-pdp-filters-button",
                        n_clicks=0,
                        className="btn btn-outline-primary w-100 shadow-sm mb-4",
                    ),
                    html.Div(
                        id="pdp-filters",
                        className="overflow-auto custom-scrollbar",
                        style={
                            "max-height": "calc(100vh - 300px)",
                            "backgroundColor": "#f0f7f0",
                            "padding": "15px",
                            "borderRadius": "8px",
                            "width": "100%",
                        },
                    ),
                ],
                id="pdp-filters-offcanvas",
                title="",
                placement="end",
                scrollable=True,
                is_open=False,
                style={"width": "300px"},
            ),
        ],
        className="container-fluid px-4 py-3",
    )


#    ____      _ _ _                _
#   / ___|__ _| | | |__   __ _  ___| | _____
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/


@callback(
    [
        Output("store-model-results", "data"),
        Output("models-toast", "is_open", allow_duplicate=True),
        Output("models-toast", "children", allow_duplicate=True),
    ],
    Input("url", "pathname"),
)
def load_model_results(pathname):
    """Load model results."""
    if pathname != "/model":
        raise dash.exceptions.PreventUpdate

    if not (helpers.FILES_PATH / "result" / "model_results.json").exists():
        return (
            dash.no_update,
            True,
            "Need to put results in the model_results.json file to display results.",
        )

    # load models
    model_results = metrics.ModelResults(filepath="model_results.json")

    return Serverside(model_results, key="model_results"), False, ""


@callback(
    [Output("pdp-selectors", "children"), Output("pdp-filters", "children")],
    Input("url", "pathname"),
    [State("store-config", "data"), State("store-dataset", "data")],
)
def load_pdp_selectors(pathname, config, dataset):
    """Load pdp selectors."""
    if pathname != "/model" or config is None:
        raise dash.exceptions.PreventUpdate

    # Add check for dataset
    if dataset is None:
        logger.warning("Dataset is None in load_pdp_selectors")
        return dash.no_update, dash.no_update

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

    pdp_filters = dh.generate_filters(
        df=dataset,
        prefix="pdp",
        config=config,
    )["filters"]

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
    scorecard = dh.flatten_columns(scorecard)
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
        "field": "model_name__",
        "headerName": "model_name__",
        "pinned": "left",
    }
    score_column_defs = dash_formats.remove_column_defs(
        score_column_defs, "model_name__"
    )
    score_column_defs.insert(0, score_col_def)
    score_column_defs = dash_formats.group_column_defs(score_column_defs)

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
        numerator=[numerator],
        denominator=[denmoninator],
    )

    return dcc.Graph(figure=target_chart)


@callback(
    [
        Output("pdp-chart", "children"),
        Output("models-toast", "is_open", allow_duplicate=True),
        Output("models-toast", "children", allow_duplicate=True),
    ],
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
        return dash.no_update, True, "Model file or x-axis column is not selected."

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
    add_constant = model_results.model[
        model_results.model["model_name"] == model_name
    ].iloc[0]["preprocess_params"]["add_constant"]
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
        add_constant=add_constant,
    )
    mapping = preprocess_dict["mapping"]
    md_encoded = preprocess_dict["md_encoded"]

    # get parameters from config
    line_color = dh._inputs_parse_id(states_info, "color_selector")
    weight = dh._inputs_parse_id(states_info, "weights_selector")
    secondary = dh._inputs_parse_id(states_info, "secondary_selector")
    x_bins = dh._inputs_parse_id(states_info, "x_bins_selector")

    # verify parameters are in the model
    if not any(
        col in mapping.keys() for col in [x_axis_col, line_color, weight, secondary]
    ):
        return dash.no_update, True, "Column was not included in model."

    # create pdp
    pdp_chart = charters.pdp(
        model=model,
        df=md_encoded,
        x_axis=x_axis_col,
        line_color=line_color,
        weight=weight,
        secondary=secondary,
        mapping=mapping,
        x_bins=x_bins,
        quick=True,
    )

    return dcc.Graph(figure=pdp_chart), False, ""


# Add new callbacks for the filter offcanvas
@callback(
    Output("pdp-filters-offcanvas", "is_open"),
    [Input("open-pdp-filters-button", "n_clicks")],
    [State("pdp-filters-offcanvas", "is_open")],
)
def toggle_pdp_filters_offcanvas(n_clicks, is_open):
    """Toggle the filters offcanvas."""
    if n_clicks:
        return not is_open
    return is_open


@callback(
    [
        Output({"type": "pdp-str-filter", "index": ALL}, "value"),
        Output({"type": "pdp-num-filter", "index": ALL}, "value"),
    ],
    [Input("reset-pdp-filters-button", "n_clicks")],
    [State("store-dataset", "data"), State("store-config", "data")],
    prevent_initial_call=True,
)
def reset_pdp_filters(n_clicks, dataset, config):
    """Reset all filters to default values."""
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    logger.debug("resetting filters")
    filter_dict = dh.generate_filters(df=dataset, prefix="pdp", config=config)
    str_reset_values = [[]] * len(filter_dict["str_cols"])
    num_reset_values = [
        [dataset[col].min(), dataset[col].max()] for col in filter_dict["num_cols"]
    ]
    return str_reset_values, num_reset_values


@callback(
    Output({"type": "pdp-collapse", "index": ALL}, "is_open"),
    Output({"type": "pdp-collapse-button", "index": ALL}, "children"),
    Input({"type": "pdp-collapse-button", "index": ALL}, "n_clicks"),
    State({"type": "pdp-collapse", "index": ALL}, "is_open"),
    State({"type": "pdp-collapse-button", "index": ALL}, "children"),
    prevent_initial_call=True,
)
def toggle_pdp_collapse(n_clicks, is_open, children):
    """Toggle collapse state of filter checklists."""
    if not n_clicks or not any(n_clicks):
        raise dash.exceptions.PreventUpdate

    # Find which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        return [False] * len(is_open), children

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    button_idx = eval(button_id)["index"]

    # Update the collapse states and button icons
    new_is_open = []
    new_children = []

    for _, (col, is_open_state, child) in enumerate(
        zip(
            [x["id"]["index"] for x in ctx.inputs_list[0]],
            is_open,
            children,
            strict=False,
        )
    ):
        # Update collapse state
        new_state = not is_open_state if col == button_idx else is_open_state
        new_is_open.append(new_state)

        # Update button content
        label = child[0]["props"]["children"]  # Get the column name
        new_children.append(
            [
                html.Span(label, style={"flex-grow": 1}),
                html.I(className=f"fas fa-chevron-{'up' if new_state else 'down'}"),
            ]
        )

    return new_is_open, new_children
