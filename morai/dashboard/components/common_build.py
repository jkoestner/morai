"""Common build structures used in dashboard components."""

import ast
from dataclasses import dataclass, field
from typing import Any

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import html
from dash_extensions.enrich import dcc

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


@dataclass
class TabSpec:
    """
    Specification for a single tab in a tabbed content area.

    Attributes
    ----------
    label : str
        Display name shown on the tab.
    tab_id : str
        Identifier suffix for the tab (e.g. "chart", "table"). The full
        component id is namespaced by page in build_tabbed_content.

    """

    label: str
    tab_id: str

    def full_id(self, prefix: str) -> str:
        """Get the full component id for this tab, namespaced by page."""
        return f"{prefix}-tab-{self.tab_id}"


@dataclass
class FilterPanel:
    """
    Specification for the filter offcanvas panel.

    Attributes
    ----------
    children : list
        Components rendered inside the offcanvas body. The page is
        responsible for the contents (bookmarks, reset button, filter
        controls, etc.).
    title : str
        Title shown in the offcanvas header.
    button_label : str
        Text on the trigger button.

    """

    children: list[Any] = field(default_factory=list)
    title: str = "Filters"
    button_label: str = "Show Filters"


def _build_tabbed_content(
    tabs: list,
    prefix: str,
    active_tab: str | None = None,
    filter_panel: FilterPanel | None = None,
    include_secondary_chart: bool = False,
) -> list:
    """
    Build a tabbed content area with optional filter button and exports.

    Notes
    -----
    - the containing Tabs component will have id "{prefix}-tabs"
    - the tab content container will have id "{prefix}-tab-content"
    - each tab will have an id of the form "{prefix}-tab-{tab_id}"
    - the offcanvas filter button will have id "{prefix}-open-offcanvas-button"

    Parameters
    ----------
    tabs : list of TabSpec
        Tab specifications in display order.
    prefix : str
        prefix to use for the ids of all components
        (tabs, content container, filter button)
    active_tab : str, optional
        tab_id of the initially active tab. Defaults to the first tab.
    filter_panel : FilterPanel, optional
        If provided, renders a trigger button and offcanvas with the
        given children. If None, no filter UI is rendered.
    include_secondary_chart : bool
        Whether to render a secondary chart container below the tabs.

    """
    # build each tab
    tab_components = [
        dbc.Tab(
            children=[],
            label=t.label,
            tab_id=f"{prefix}-tab-{t.tab_id}",
            label_class_name="fw-bold",
            active_label_class_name="text-primary",
        )
        for t in tabs
    ]

    # set active tab (default to first if not specified)
    default_active = active_tab or tabs[0].full_id(prefix)

    # add filter button if needed, then tabs, then exports
    content = []
    if filter_panel is not None:
        content.append(
            dbc.Button(
                [html.I(className="fas fa-filter me-2"), filter_panel.button_label],
                id={"type": "open-offcanvas-button", "prefix": prefix},
                className="mb-3",
                color="primary",
                style={"width": "auto"},
            )
        )
    content.append(
        dbc.Tabs(
            tab_components,
            id=f"{prefix}-tabs",
            active_tab=default_active,
            className="mb-3",
        )
    )

    # add loaders and optional secondary chart
    loaders = [
        dcc.Loading(
            id=f"{prefix}-loading-tab-content",
            custom_spinner=dmc.Skeleton(visible=True, h="100%", w="100%"),
            children=html.Div(
                id=f"{prefix}-tab-content",
                className="bg-white rounded-3 shadow-sm p-4 border border-light",
            ),
        ),
    ]
    if include_secondary_chart:
        loaders.append(
            dcc.Loading(
                id=f"{prefix}-loading-chart-secondary",
                custom_spinner=dmc.Skeleton(visible=True, h="100%", w="100%"),
                children=html.Div(
                    id=f"{prefix}-chart-secondary",
                    className=(
                        "mt-4 bg-white rounded-3 shadow-sm p-4 border border-light"
                    ),
                ),
            )
        )
    content.append(html.Div(loaders, className="h-100"))

    # filter offcanvas
    if filter_panel is not None:
        content.append(
            dbc.Offcanvas(
                filter_panel.children,
                id={"type": "filters-offcanvas", "prefix": prefix},
                title=filter_panel.title,
                placement="end",
                scrollable=True,
                is_open=False,
                className="offcanvas",
            )
        )

    return content


def _build_export_button(tab: str, page: str) -> html.Button:
    """
    Build an export button.

    Parameters
    ----------
    tab : str
        The tab to create the button for
    page : page
        The page to create the button for

    Returns
    -------
    export_button : html.Button
        An html export button

    """
    return html.Button(
        [html.I(className="fas fa-download me-2"), "Export to CSV"],
        id={"type": "export-button", "tab": tab, "page": page},
        className="btn btn-primary mt-2 mb-2",
        style={"display": "inline-block"},
    )


def register_export_callback(app) -> None:
    """
    Register a universal callback for exporting table data to CSV.

    This function should be called once in the app initialization to register
    the export functionality for all data tables across the application.

    Parameters
    ----------
    app : dash.Dash
        The Dash application instance.

    Notes
    -----
    For this callback to work, the following components must be present:
    1. A Download component with id="download-dataframe-csv" in each page's layout
    2. Export buttons with pattern-matching ID:
       {"type": "export-button", "tab": <tab_name>, "page": <page_name>}
    3. Data tables with pattern-matching ID:
       {"type": "data-table", "tab": <tab_name>, "page": <page_name>}

    The tab and page values must match between the button and table for proper pairing.

    """
    import dash  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from dash_extensions.enrich import (  # noqa: PLC0415
        ALL,
        Input,
        Output,
        State,
        callback,
        callback_context,
        dcc,
    )

    @callback(
        Output("download-dataframe-csv", "data"),
        Input({"type": "export-button", "tab": ALL, "page": ALL}, "n_clicks"),
        State({"type": "data-table", "tab": ALL, "page": ALL}, "rowData"),
        prevent_initial_call=True,
    )
    def export_table(
        n_clicks_list: list[int | None], table_data_list: list[list[Any]]
    ) -> None:
        """
        Export table data to CSV.

        This generic function handles exporting data from any
        table with an export button.

        The button and table must use pattern-matching IDs
        with the following structure:
        - Button:
          {"type": "export-button", "tab": <tab_name>, "page": <page_name>}
        - Table:
          {"type": "data-table", "tab": <tab_name>, "page": <page_name>}

        Where <tab_name> identifies the specific tab
        and <page_name> identifies the page.
        """
        ctx = callback_context
        if not ctx.triggered or not ctx.triggered[0]["value"]:
            return dash.no_update

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        button_id = ast.literal_eval(triggered_id)
        tab = button_id["tab"]
        page = button_id["page"]

        # Find the matching table data by comparing both tab and page values
        for i, table_data in enumerate(table_data_list):
            if not table_data:
                continue

            # Get the corresponding table ID
            table_id = ctx.states_list[0][i]["id"]

            # Check if this table matches the clicked button's tab and page
            if table_id["tab"] == tab and table_id["page"] == page:
                df = pd.DataFrame(table_data)
                filename = f"{page}_{tab}.csv"
                return dcc.send_data_frame(df.to_csv, filename, index=False)

        return dash.no_update


def register_filter_callbacks(app) -> None:
    """
    Register universal filter behavior callbacks.

    This function should be called once during app initialization to register:
    - Filter offcanvas toggle (for any page with the filter button)
    - Filter checklist collapse/expand
    - Filter reset

    For these callbacks to work, components must use pattern-matching IDs:

    Offcanvas toggle:
    - Button: id="{prefix}-open-offcanvas-button"
    - Offcanvas: id="{prefix}-filters-offcanvas"

    Collapse:
    - Collapse button:
      {"type": "filter-collapse-button", "prefix": <prefix>, "index": <col>}
    - Collapse container:
      {"type": "filter-collapse", "prefix": <prefix>, "index": <col>}

    Reset:
    - Reset button: {"type": "filter-reset-button", "prefix": <prefix>}
    - String filters: {"type": "filter-str", "prefix": <prefix>, "index": <col>}
    - Numeric filters: {"type": "filter-num", "prefix": <prefix>, "index": <col>}
    - Initial values store: id="{prefix}-store-initial-filters" with shape:
      {"str_cols": [...], "num_cols": [...], "min_max": {"col_min": x, "col_max": y}}
    """
    import dash  # noqa: PLC0415
    from dash_extensions.enrich import (  # noqa: PLC0415
        ALL,
        MATCH,
        Input,
        Output,
        State,
        callback,
        callback_context,
    )

    # offcanvas toggle
    @callback(
        Output({"type": "filters-offcanvas", "prefix": MATCH}, "is_open"),
        Input({"type": "open-offcanvas-button", "prefix": MATCH}, "n_clicks"),
        State({"type": "filters-offcanvas", "prefix": MATCH}, "is_open"),
        prevent_initial_call=True,
    )
    def toggle_filters_offcanvas(n_clicks, is_open):
        """Toggle the filters offcanvas."""
        if n_clicks:
            return not is_open
        return is_open

    # collapse checklists
    @callback(
        Output({"type": "filter-collapse", "prefix": MATCH, "index": ALL}, "is_open"),
        Output(
            {"type": "filter-collapse-button", "prefix": MATCH, "index": ALL},
            "children",
        ),
        Input(
            {"type": "filter-collapse-button", "prefix": MATCH, "index": ALL},
            "n_clicks",
        ),
        State({"type": "filter-collapse", "prefix": MATCH, "index": ALL}, "is_open"),
        State(
            {"type": "filter-collapse-button", "prefix": MATCH, "index": ALL},
            "children",
        ),
        prevent_initial_call=True,
    )
    def toggle_collapse(n_clicks, is_open, children):
        """Toggle collapse state of filter checklists."""
        if not n_clicks or not any(n_clicks):
            raise dash.exceptions.PreventUpdate

        ctx = callback_context
        if not ctx.triggered:
            return [False] * len(is_open), children

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        button_idx = ast.literal_eval(button_id)["index"]

        new_is_open = []
        new_children = []
        for col, is_open_state, child in zip(
            [x["id"]["index"] for x in ctx.inputs_list[0]],
            is_open,
            children,
            strict=False,
        ):
            new_state = not is_open_state if col == button_idx else is_open_state
            new_is_open.append(new_state)
            label = child[0]["props"]["children"]
            new_children.append(
                [
                    html.Span(label, style={"flex-grow": 1}),
                    html.I(className=f"fas fa-chevron-{'up' if new_state else 'down'}"),
                ]
            )
        return new_is_open, new_children
