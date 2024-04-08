"""Explore dashboard."""

import dash
import polars as pl
from dash import Input, Output, callback, dcc, html

from morai.dashboard.components import sidebars
from morai.dashboard.utils import dashboard_helper as dh
from morai.experience import charters
from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)

dash.register_page(__name__, path="/explore", title="morai - Explore")

#   ____        _          _                    _
#  |  _ \  __ _| |_ __ _  | |    ___   __ _  __| |
#  | | | |/ _` | __/ _` | | |   / _ \ / _` |/ _` |
#  | |_| | (_| | || (_| | | |__| (_) | (_| | (_| |
#  |____/ \__,_|\__\__,_| |_____\___/ \__,_|\__,_|

# load config
config = dh.load_config()
config_dataset = config["datasets"][config["general"]["dataset"]]
# load dataset
file_path = helpers.FILES_PATH / "dataset" / config_dataset["filename"]
if file_path.suffix == ".parquet":
    pl.enable_string_cache()
    lzdf = pl.scan_parquet(file_path)
    mortality_df = lzdf.collect()
    mortality_df = mortality_df.to_pandas()


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
            dcc.Store(id="user-config", storage_type="session"),
            dcc.Location(id="explore-url", refresh=False),
            # -----------------------------------------------------
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
                            html.P(
                                [
                                    "Dataset: ",
                                    html.Span(
                                        f"{config_dataset['filename']}",
                                        style={"fontWeight": "bold"},
                                    ),
                                ],
                            ),
                            html.H5(
                                "Frequency Number",
                                id="section-freq-num",
                                className="bg-secondary text-white p-2 mb-2",
                            ),
                            dcc.Loading(
                                id="loading-content",
                                type="dot",
                                children=html.Div(id="chart-freq-num"),
                            ),
                            html.H5(
                                "Frequency Categorical",
                                id="section-freq-cat",
                                className="bg-secondary text-white p-2 mb-2",
                            ),
                            dcc.Loading(
                                id="loading-content",
                                type="dot",
                                children=html.Div(id="chart-freq-cat"),
                            ),
                            html.H5(
                                "Target",
                                id="section-target",
                                className="bg-secondary text-white p-2 mb-2",
                            ),
                            dcc.Loading(
                                id="loading-content",
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
    Input("explore-url", "hash"),
    prevent_initial_call=True,
)
def set_active_link(hash):
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
    Input("explore-url", "url"),
)
def update_charts(hash):
    """Update charts when page is loaded."""
    # charts
    chart_freq_num = charters.frequency(
        mortality_df,
        cols=3,
        features=[
            "observation_year",
            "issue_age",
            "duration",
            "issue_year",
            "attained_age",
        ],
        sum_var="amount_exposed",
    )

    chart_freq_cat = charters.frequency(mortality_df, cols=3, sum_var="amount_exposed")

    chart_target = charters.target(
        df=mortality_df,
        target="risk",
        cols=3,
        features=[
            "observation_year",
            "sex",
            "smoker_status",
            "insurance_plan",
            "issue_age",
            "duration",
            "face_amount_band",
            "issue_year",
            "attained_age",
            "soa_post_lvl_ind",
            "number_of_pfd_classes",
            "preferred_class",
        ],
        numerator="death_claim_amount",
        denominator="amount_exposed",
    )

    chart_freq_num = dcc.Graph(figure=chart_freq_num)
    chart_freq_cat = dcc.Graph(figure=chart_freq_cat)
    chart_target = dcc.Graph(figure=chart_target)

    return chart_freq_num, chart_freq_cat, chart_target
