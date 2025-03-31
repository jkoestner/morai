"""Components - dash formats."""

from morai.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


def get_column_defs(table):
    """
    Get the column definitions.

    Parameters
    ----------
    table : pd.DataFrame
        The table

    Returns
    -------
    column_defs : list
        The column definitions

    """
    column_defs = []
    for col in table.columns:
        # specific case for columns
        if any(
            substring in col.lower()
            for substring in ["smape", "vals", "table_1", "table_2"]
        ):
            column_defs.append(
                {
                    "field": col,
                    "headerName": col,
                    "valueFormatter": {"function": "d3.format(',.5f')(params.value)"},
                }
            )
        # update integer columns
        elif table[col].dtype.kind in "i":
            if "year" in col.lower():
                column_defs.append(
                    {
                        "field": col,
                        "headerName": col,
                        "valueFormatter": {"function": "d3.format('d')(params.value)"},
                    }
                )
            else:
                column_defs.append(
                    {
                        "field": col,
                        "headerName": col,
                        "valueFormatter": {"function": "d3.format(',')(params.value)"},
                    }
                )
        # update float columns
        elif table[col].dtype.kind in "f":
            if any(
                substring in col.lower()
                for substring in ["ae", "pct", "ratio", "risk", "r2_score"]
            ):
                column_defs.append(
                    {
                        "field": col,
                        "headerName": col,
                        "valueFormatter": {
                            "function": "d3.format('.2%')(params.value)"
                        },
                    }
                )
            else:
                column_defs.append(
                    {
                        "field": col,
                        "headerName": col,
                        "valueFormatter": {
                            "function": "d3.format(',.1f')(params.value)"
                        },
                    }
                )
        # update other types
        else:
            column_defs.append(
                {
                    "field": col,
                    "headerName": col,
                }
            )
    return column_defs


def remove_column_defs(column_defs, col_name):
    """
    Remove a column from the column definitions.

    Parameters
    ----------
    column_defs : list
        The column definitions
    col_name : str
        The column name to remove

    Returns
    -------
    column_defs : list
        The column definitions

    """
    for col in column_defs:
        if col["field"] == col_name:
            column_defs.remove(col)
    return column_defs


def group_column_defs(column_defs):
    """
    Create column groups for the column definitions.

    This is based on the first part of the string before the first underscore.
    e.g. 'train_ae' and 'train_r2_score' would be grouped under 'train'.

    Parameters
    ----------
    column_defs : list
        The column definitions

    Returns
    -------
    group_column_defs : list
        The column definitions

    """
    groups = {}

    # parse the column definitions to group them by prefix
    for col_def in column_defs:
        field = col_def["field"]
        value_formatter = col_def.get("valueFormatter", None)
        pinned = col_def.get("pinned", None)

        # split by double underscore to get the prefix and metric
        parts = field.split("__")
        if len(parts) > 1:
            prefix = parts[0]
            metric = "__".join(parts[1:])
        else:
            prefix = "general"
            metric = parts[0]

        if prefix not in groups:
            groups[prefix] = []

        # create the metric definition
        child_def = {"field": field, "headerName": metric}
        if value_formatter:
            child_def["valueFormatter"] = value_formatter
        if pinned:
            child_def["pinned"] = pinned

        groups[prefix].append(child_def)

    # combine the group and child definitions
    group_column_defs = []
    for group, children in groups.items():
        group_def = {
            "headerName": group,
            "marryChildren": True,
            "headerClass": "center-aligned-group-header",
            "suppressStickyLabel": True,
            "children": children,
        }
        group_column_defs.append(group_def)

    return group_column_defs
