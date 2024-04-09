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
    ColumnDefs : list
        The column definitions

    """
    columnDefs = []
    for col in table.columns:
        if table[col].dtype.kind in "i":
            if "year" in col.lower():
                columnDefs.append(
                    {
                        "field": col,
                        "headerName": col,
                        "valueFormatter": {"function": "d3.format('d')(params.value)"},
                    }
                )
            else:
                columnDefs.append(
                    {
                        "field": col,
                        "headerName": col,
                        "valueFormatter": {"function": "d3.format(',')(params.value)"},
                    }
                )
        elif table[col].dtype.kind in "f":
            if col in ["ratio", "risk", "ae"] or "pct" in col.lower():
                columnDefs.append(
                    {
                        "field": col,
                        "headerName": col,
                        "valueFormatter": {
                            "function": "d3.format('.2%')(params.value)"
                        },
                    }
                )
            else:
                columnDefs.append(
                    {
                        "field": col,
                        "headerName": col,
                        "valueFormatter": {
                            "function": "d3.format(',.1f')(params.value)"
                        },
                    }
                )
        else:
            columnDefs.append(
                {
                    "field": col,
                    "headerName": col,
                }
            )
    return columnDefs
