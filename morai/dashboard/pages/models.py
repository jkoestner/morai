"""Experience dashboard."""

import dash
import dash_ag_grid as dag
from dash import (
    dcc,
    html,
)

from morai.dashboard.components import dash_formats
from morai.forecast import metrics
from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


dash.register_page(__name__, path="/model", title="morai - Model")

# load models
model_results = metrics.ModelResults(filepath="model_results.json")
models = model_results.model
scorecard = model_results.scorecard


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
            dcc.Store(id="user-config", storage_type="session"),
            # -----------------------------------------------------
            html.H4(
                "Model Analysis",
                className="bg-primary text-white p-2 mb-2 text-center",
            ),
            dag.AgGrid(
                rowData=models.to_dict("records"),
                columnDefs=dash_formats.get_column_defs(models),
            ),
            dag.AgGrid(
                rowData=scorecard.to_dict("records"),
                columnDefs=dash_formats.get_column_defs(scorecard),
            ),
        ],
        className="container",
    )


#    ____      _ _ _                _
#   / ___|__ _| | | |__   __ _  ___| | _____
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/
