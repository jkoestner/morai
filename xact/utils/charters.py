"""Collection of visualization tools."""
import logging
import logging.config
import math
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xact.utils import helpers

logging.config.fileConfig(
    os.path.join(helpers.ROOT_PATH, "logging.ini"),
)
logger = logging.getLogger(__name__)


def chart(
    data,
    x_axis,
    y_axis=None,
    color=None,
    type="line",
    actual=None,
    expected=None,
    exposure=None,
    title=None,
    **kwargs,
):
    """
    Create a chart with Plotly Express.

    Parameters
    ----------
    data : pd.DataFrame
        The data to chart.
    x_axis : str
        The column name to use for the x-axis.
    y_axis : str
        The column name to use for the y-axis.
    color : str, optional (default=None)
        The column name to use for the color.
    type : str, optional (default="line")
        The type of chart to create. Options are "line" or "bar".
    actual : str
        The column name to use for the actual values.
    expected : str
        The column name to use for the expected values.
    exposure : str
        The column name to use for the exposure values.
    title : str
        The title of the chart.
    **kwargs : dict
        Additional keyword arguments to pass to Plotly Express.

    Returns
    -------
    fig : Figure
        The chart
    """
    # heatmap sum values are color rather than y_axis
    if type == "heatmap":
        if not color:
            raise ValueError("Color parameter is required for heatmap.")
        _y_axis = y_axis
        _color = color
        color = _y_axis
        y_axis = _color

    # getting the columns to sum by
    sum_col_dict = {"ae": [actual, expected], "qx": [actual, exposure]}
    sum_cols = sum_col_dict.get(y_axis, y_axis)

    # groupby by the x_axis and color
    groupby_cols = [x_axis]
    if color:
        groupby_cols.append(color)

    grouped_data = (
        data.groupby(groupby_cols, observed=True)[sum_cols].sum().reset_index()
    )

    # calculating fields
    if y_axis in sum_col_dict:
        actual, expected = sum_col_dict[y_axis]
        grouped_data[y_axis] = grouped_data[actual] / grouped_data[expected]

    # Selecting the plot type based on the 'chart_type' parameter
    if type == "line":
        if not title:
            title = f"{y_axis} by {x_axis} and {color}"
        fig = px.line(
            grouped_data,
            x=x_axis,
            y=y_axis,
            color=color,
            title=title,
            **kwargs,
        )
    elif type == "bar":
        if not title:
            title = f"{y_axis} by {x_axis} and {color}"
        fig = px.bar(
            grouped_data,
            x=x_axis,
            y=y_axis,
            color=color,
            title=title,
            **kwargs,
        )
    elif type == "heatmap":
        heatmap_data = grouped_data.pivot(index=_y_axis, columns=x_axis, values=_color)
        if not title:
            title = f"Heatmap of {_color} by {x_axis} and {_y_axis}"
        fig = px.imshow(
            heatmap_data,
            labels={"x": x_axis, "y": _y_axis, "color": _color},
            title=title,
            **kwargs,
        )
    else:
        raise ValueError("Unsupported type. Use 'line', 'heatmap', or 'bar'.")

    return fig


def compare_rates(
    df,
    variable,
    rates,
    weights="amount_exposed",
    secondary="policies_exposed",
):
    """
    Compare rates by a variable.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use.
    variable : str
        The column name for the variable to compare by.
    rates : list
        The column names for the rates to compare.
    weights : str (default="amount_exposed")
        The column name for the weights to use.
    secondary : str (default="policies_exposed")
        The column name for the secondary y-axis.

    Returns
    -------
    fig : Figure
        The chart
    """
    # Group and calculate weighted means
    grouped_df = (
        df.groupby([variable], observed=True)
        .apply(
            lambda x: pd.Series(
                {
                    **{
                        rate: _weighted_mean(x[rate], weights=x[weights])
                        if weights
                        else x[rate].mean()
                        for rate in rates
                    },
                    **{secondary: x[secondary].sum() if secondary else None},  # noqa: PIE800
                }
            )
        )
        .reset_index()
    )

    # Create a subplot with a secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the bar chart for policies_exposed first (background)
    fig.add_trace(
        go.Bar(
            x=grouped_df[variable],
            y=grouped_df[secondary],
            name=secondary,
            marker_color="rgba(135, 206, 250, 0.6)",
        ),
        secondary_y=True,
    )

    # Add the line charts for rates (foreground)
    for rate in rates:
        fig.add_trace(
            go.Scatter(
                x=grouped_df[variable],
                y=grouped_df[rate],
                name=rate,
                mode="lines+markers",
            ),
            secondary_y=False,
        )

    # Set plot layout
    fig.update_layout(title_text=f"Comparison of {rates} by {variable}")
    fig.update_xaxes(title_text=variable)
    fig.update_yaxes(title_text="Rates", secondary_y=False)
    fig.update_yaxes(title_text=secondary, secondary_y=True)

    return fig


def histogram(df, cols=1, features=None, sum_var=None):
    """
    Generate histogram plots.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use.
    cols : int, optional (default=1)
        The number of columns to use for the subplots.
    features : list, optional (default=None)
        The features to use for the plot. Default is to use all categorical features.
    sum_var : str, optional (default=None)
        The column name to use for the sum. Default is to use frequency.

    Returns
    -------
    fig : Figure
        The chart
    """
    # get the features for the plot
    if features is None:
        features = df.select_dtypes(exclude=[np.number]).columns.to_list()
    if sum_var is None:
        sum_var = "_count"
        df["_count"] = 1

    # creating the plot grid
    rows = len(features) // cols + (len(features) % cols > 0)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=features)

    # create histogram plot for each feature
    for i, col in enumerate(features, start=1):
        frequency = df.groupby(col, observed=True)[sum_var].sum().reset_index()
        frequency.columns = [col, sum_var]
        frequency = frequency.sort_values(by=col, ascending=True)

        # create the subplot
        row = math.ceil(i / cols)
        col_num = (i + cols - 1) % cols + 1
        for trace in px.bar(frequency, x=col, y=sum_var).data:
            fig.add_trace(trace, row=row, col=col_num)

    # update the layout
    fig.update_layout(
        height=300 * rows,
        showlegend=False,
        title_text=f"Histograms of variables using {sum_var}",
    )

    return fig


def pdp(model, X, feature, mapping=None):
    """
    Create a partial dependence plot (PDP) for the DataFrame.

    Parameters
    ----------
    model : model
        The model to use.
    X : pd.DataFrame
        The features in the model.
    feature : str
        The feature to create the PDP for.
    mapping : dict, optional (default=None)
        The mapping to use for the feature.

    Returns
    -------
    fig : Figure
        The chart
    """
    logger.info(f"Creating partial dependence plot for {feature}")
    # get values that variable has
    if mapping and feature in mapping:
        feature_values = list(mapping[feature].values())
    else:
        feature_values = np.linspace(X[feature].min(), X[feature].max(), 100)

    # average value of features
    X_avg = X.mean()

    # calculate predictions
    preds = []
    for value in feature_values:
        X_temp = X_avg.copy()
        X_temp[feature] = value
        # Ensure X_temp is a DataFrame
        if isinstance(X_temp, pd.Series):
            X_temp = X_temp.to_frame().T
        # X_temp["const"] = 1
        pred = model.predict(X_temp)[0]
        preds.append(pred)

    mean_pred = np.mean(preds)
    percent_diff_from_mean = [(p - mean_pred) / mean_pred + 1 for p in preds]
    plot_df = pd.DataFrame({feature: feature_values, "%_diff": percent_diff_from_mean})

    # use mapping to get the original feature values
    if mapping and feature in mapping:
        reversed_mapping = {v: k for k, v in mapping[feature].items()}
        plot_df[feature] = plot_df[feature].map(reversed_mapping)

    # Create the plot using Plotly Express

    fig = px.line(
        plot_df,
        x=feature,
        y="%_diff",
        title="Partial Dependency Plot",
        labels={feature: feature, "%_diff": "%_diff"},
    )
    fig.update_layout(yaxis_tickformat=".1%")

    return fig


def scatter(df, target, numeric=True, sample_nbr=100, cols=3):
    """
    Create scatter plots for the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use.
    target : str
        The target variable.
    numeric : bool, optional
        Whether to use numeric features.
    sample_nbr : float, optional
        The sample amount to use.
    cols : int, optional
        The number of columns to use for the subplots.

    Returns
    -------
    fig : Figure
        The chart
    """
    if sample_nbr:
        sample_amt = sample_nbr / len(df)
        df = df.sample(frac=sample_amt)
    # Finding features
    numeric_dtypes = ["int16", "int32", "int64", "float16", "float32", "float64"]
    if numeric:
        features = [i for i in df.columns if df[i].dtype in numeric_dtypes]
    else:
        features = [i for i in df.columns if df[i].dtype not in numeric_dtypes]

    # Number of rows for the subplot grid
    num_plots = len(features)
    num_rows = math.ceil(num_plots / cols)

    # Create a subplot grid
    fig = make_subplots(rows=num_rows, cols=cols, subplot_titles=features)

    # Create each scatter plot
    for i, feature in enumerate(features, 1):
        row = (i - 1) // cols + 1
        col = (i - 1) % cols + 1

        fig.add_trace(
            go.Scattergl(x=df[feature], y=df[target], mode="markers", name=feature),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(height=300 * num_rows, title_text="Scatter Plots")

    return fig


def target(df, features=None, target=None, cols=3):
    """
    Create multiplot showing variable relationship with target.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use.
    features : list, optional
        The features to use for the plot. Default is to use all features.
    target : str
        The target variable.
    cols : int, optional
        The number of columns to use for the subplots.

    Returns
    -------
    fig : Figure
        The chart
    """
    if features is None:
        features = df.columns
    actual = "death_claim_amount"
    exposure = "amount_exposed"
    target = "target"

    # Number of rows for the subplot grid
    num_plots = len(features)
    num_rows = math.ceil(num_plots / cols)

    # Create a subplot grid
    fig = make_subplots(rows=num_rows, cols=cols, subplot_titles=features)

    # Create each plot
    for i, feature in enumerate(features, 1):
        row = (i - 1) // cols + 1
        col = (i - 1) % cols + 1

        grouped_data = (
            df.groupby(feature, observed=True)[[actual, exposure]].sum().reset_index()
        )
        grouped_data["target"] = grouped_data[actual] / grouped_data[exposure]
        fig.add_trace(
            go.Scatter(
                x=grouped_data[feature],
                y=grouped_data[target],
                mode="lines",
                name=feature,
            ),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(height=300 * num_rows, title_text="Target Plots")

    return fig


def _weighted_mean(values, weights):
    """
    Calculate the weighted mean.

    Parameters
    ----------
    values : list
        The values to use.
    weights : list
        The weights to use.

    Returns
    -------
    weighted_mean : float
        The weighted mean
    """
    if np.sum(weights) == 0:
        return np.nan
    return np.average(values, weights=weights)
