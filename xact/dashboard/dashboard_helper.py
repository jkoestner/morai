"""
Utilities for app.

Provides functions for common used functions in app.
"""

from dash import dcc, html


def get_menu():
    """Provide menu for pages."""
    menu = html.Div(
        [
            dcc.Link(
                "Experience   ",
                href="/experience",
            ),
        ]
    )
    return menu


def convert_to_short_number(number):
    """
    Convert number to short number.

    Parameters
    ----------
    number : float
        Number to convert to short number.

    Returns
    -------
    str
        Short number.

    """
    for unit in ["", "K", "M", "B", "T"]:
        if abs(number) < 1000:
            return f"{number:.1f}{unit}"
        number /= 1000
    return f"{number:.1f}T"


def generate_filters(df):
    """
    Generate dropdown options.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to generate dropdown options.

    Returns
    -------
    filters : list
        List of dropdowns.
    str_cols : list
        List of string columns.
    num_cols : list
        List of numeric columns.

    """
    filters = []
    str_cols = []
    num_cols = []
    for col in df.columns:
        if isinstance(df[col][0], str):
            filter = dcc.Dropdown(
                id={"type": "str-filter", "index": col},
                options=[{"label": i, "value": i} for i in sorted(df[col].unique())],
                multi=True,
                placeholder=f"Select {col}",
            )
            str_cols.append(col)
        elif df[col].nunique() < 10:
            filter = dcc.Dropdown(
                id={"type": "str-filter", "index": col},
                options=[{"label": i, "value": i} for i in sorted(df[col].unique())],
                multi=True,
                placeholder=f"Select {col}",
            )
            str_cols.append(col)
        else:
            filter = dcc.RangeSlider(
                id={"type": "num-filter", "index": col},
                min=df[col].min(),
                max=df[col].max(),
                marks=None,
                value=[df[col].min(), df[col].max()],
                tooltip={"always_visible": True, "placement": "bottom"},
                className="p-0",
            )
            num_cols.append(col)
        filters.append(html.Label(col))
        filters.append(filter)
        filter_dict = {"filters": filters, "str_cols": str_cols, "num_cols": num_cols}

    return filter_dict
