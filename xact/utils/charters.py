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
    numerator=None,
    denominator=None,
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
        "ratio" is a special value that will calculate the ratio of two columns.
    color : str, optional (default=None)
        The column name to use for the color.
    type : str, optional (default="line")
        The type of chart to create. Options are "line" or "bar".
    numerator : str, optional (default=None)
        The column name to use for the numerator values.
    denominator : str, optional (default=None)
        The column name to use for the expected values.
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
    # columns will be y_axis is not the special "ratio" values
    sum_col_dict = {"ratio": [numerator, denominator]}
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
        numerator, denominator = sum_col_dict[y_axis]
        grouped_data[y_axis] = grouped_data[numerator] / grouped_data[denominator]

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
    weights=None,
    secondary=None,
):
    """
    Compare rates by a variable.

    When using qx the weight should be the exposure.
    When using ae the weight should be the expected.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use.
    variable : str
        The name for the column to compare by.
    rates : list
        A list of rates to compare
    weights : list, optional (default=None)
        A list of weights to weight the rates by.
    secondary : str, optional (default=None)
        The name of the column to have a secondary y-axis for.

    Returns
    -------
    fig : Figure
        The chart

    """
    # Checking if weights list length matches the rates list
    if weights is not None and len(rates) != len(weights):
        logger.info(
            f"The weights list is {len(weights)} long and should "
            f"be {len(rates)} long. Using the first weight for all weights."
        )
        weights = [weights[0]] * len(rates)
    # Group and calculate weighted means
    grouped_df = (
        df.groupby([variable], observed=True)
        .apply(
            lambda x: pd.Series(
                {
                    **{
                        rate: helpers._weighted_mean(x[rate], weights=x[weight])
                        if weight
                        else x[rate].mean()
                        for rate, weight in zip(
                            rates, [None] * len(rates) if weights is None else weights
                        )
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


def pdp(model, X, feature, mapping=None, weights=None):
    """
    Create a partial dependence plot (PDP) for the DataFrame.

    reference: https://christophm.github.io/interpretable-ml-book/pdp.html

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
    weights : str, optional (default=None)
        The weights to use for the feature.

    Returns
    -------
    fig : Figure
        The chart

    """
    feature_type = "passthrough"

    # extract the feature values
    if mapping and feature in mapping:
        feature_type = mapping[feature]["type"]
        if feature_type == "ohe":
            # ohe values are in the suffix
            feature_cols = [col for col in X.columns if col.startswith(feature + "_")]
            feature_values = [col[len(feature + "_") :] for col in feature_cols]
        else:
            feature_values = list(mapping[feature]["values"].values())
    # if the feature is not OHE or in the mapping, use the min and max values
    else:
        feature_values = np.linspace(X[feature].min(), X[feature].max(), 100)
    logger.info(
        f"Creating partial dependence plot for [{feature}] type: [{feature_type}]"
    )

    # average value of features
    X_avg = X.apply(lambda x: helpers._weighted_mean(x, weights=weights))
    if isinstance(X_avg.dtype, pd.SparseDtype):
        X_avg = X_avg.sparse.to_dense()

    # calculate predictions of feature by looping through the feature values
    # and using the average of the other features
    preds = []
    for value in feature_values:
        X_temp = X_avg.copy()
        if feature_type == "ohe":
            # reset all OHE columns to 0 and make the target OHE column 1
            for col in feature_cols:
                X_temp[col] = 0
            X_temp[feature + "_" + value] = 1
        else:
            X_temp[feature] = value

        # ensure X_temp is a DataFrame
        if isinstance(X_temp, pd.Series):
            X_temp = X_temp.to_frame().T

        pred = model.predict(X_temp)[0]
        preds.append(pred)

    mean_pred = np.mean(preds)
    percent_diff_from_mean = [(p - mean_pred) / mean_pred + 1 for p in preds]
    plot_df = pd.DataFrame({feature: feature_values, "%_diff": percent_diff_from_mean})

    # use mapping to get the original feature values
    if mapping and feature in mapping and feature_type != "ohe":
        reversed_mapping = {v: k for k, v in mapping[feature]["values"].items()}
        plot_df[feature] = plot_df[feature].map(reversed_mapping)

    plot_df = plot_df.sort_values(by=feature)

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


def scatter(df, target, features, sample_nbr=100, cols=3):
    """
    Create scatter plots for the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use.
    target : str
        The target variable.
    features : list
        Features to create scatter plots for.
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


def target(df, target, features=None, cols=3, numerator=None, denominator=None):
    """
    Create multiplot showing variable relationship with target.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use.
    target : str
        The target variable.
    features : list, optional
        The features to use for the plot. Default is to use all features.
    cols : int, optional
        The number of columns to use for the subplots.
    numerator : str, optional
        The column name to use for the numerator values.
    denominator : str, optional
        The column name to use for the expected values.

    Returns
    -------
    fig : Figure
        The chart

    """
    if target != "ratio" and target not in df.columns:
        raise ValueError(
            f"Target '{target}' needs to be in DataFrame columns or 'ratio'"
        )
    # features to use for the plot
    if features is None:
        features = df.columns

    # Number of rows for the subplot grid
    num_plots = len(features)
    num_rows = math.ceil(num_plots / cols)

    # Create a subplot grid
    fig = make_subplots(rows=num_rows, cols=cols, subplot_titles=features)

    # Create each plot
    for i, feature in enumerate(features, 1):
        row = (i - 1) // cols + 1
        col = (i - 1) % cols + 1

        if target == "ratio":
            grouped_data = (
                df.groupby(feature, observed=True)[[numerator, denominator]]
                .sum()
                .reset_index()
            )
            grouped_data[target] = grouped_data[numerator] / grouped_data[denominator]
        else:
            grouped_data = (
                df.groupby(feature, observed=True)[target].mean().reset_index()
            )
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
