"""Collection of visualization tools."""
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from morai.experience import experience
from morai.forecast import preprocessors
from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)

# default chart height and width used for multiplots
chart_height = 300
chart_width = 1000


def chart(
    df,
    x_axis,
    y_axis=None,
    color=None,
    type="line",
    numerator=None,
    denominator=None,
    title=None,
    sort_y=False,
    x_bins=None,
    display=True,
    **kwargs,
):
    """
    Create a chart with Plotly Express.

    This is a wrapper around a number of plotly express functions to create
    charts easily with simple parameters. The function expects the data to be
    in row/column format and can't do multiple columns (see `compare_rates` for
    this functionality).

    The function will also allow charting ratios when using numerator and
    denominator paramters.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to chart.
    x_axis : str
        The column name to use for the x-axis.
    y_axis : str
        The column name to use for the y-axis.
        "ratio" is a special value that will calculate the ratio of two columns.
    color : str, optional (default=None)
        The column name to use for the color.
    type : str, optional (default="line")
        The type of chart to create. Options are "line", "heatmap", or "bar".
    numerator : str, optional (default=None)
        The column name to use for the numerator values.
    denominator : str, optional (default=None)
        The column name to use for the expected values.
    title : str
        The title of the chart.
    sort_y : bool, optional (default=False)
        Sort the records by the y-axis descending.
    x_bins : int, optional (default=None)
        The number of bins to use for the x-axis.
    display : bool, optional (default=True)
        Whether to display figure or not.
    **kwargs : dict
        Additional keyword arguments to pass to Plotly Express.

    Returns
    -------
    fig : Figure
        The chart

    """
    df = df.copy()
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

    if x_bins:
        logger.info(f"Binning feature: [{x_axis}] with {x_bins} bins")
        df[x_axis] = preprocessors.bin_feature(df[x_axis], x_bins)

    grouped_data = df.groupby(groupby_cols, observed=True)[sum_cols].sum().reset_index()

    # calculating fields
    if y_axis in sum_col_dict:
        numerator, denominator = sum_col_dict[y_axis]
        grouped_data[y_axis] = grouped_data[numerator] / grouped_data[denominator]

    if sort_y:
        grouped_data = grouped_data.sort_values(by=y_axis, ascending=False)

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
        grouped_data = grouped_data.pivot(index=_y_axis, columns=x_axis, values=_color)
        if not title:
            title = f"Heatmap of {_color} by {x_axis} and {_y_axis}"
        fig = px.imshow(
            grouped_data,
            labels={"x": x_axis, "y": _y_axis, "color": _color},
            title=title,
            **kwargs,
        )
    else:
        raise ValueError("Unsupported type. Use 'line', 'heatmap', or 'bar'.")

    # return the dataframe instead of figure
    if not display:
        fig = grouped_data

    return fig


def compare_rates(
    df,
    x_axis,
    rates,
    line_feature=None,
    weights=None,
    secondary=None,
    log_y=False,
    x_bins=None,
    display=True,
    **kwargs,
):
    """
    Compare rates by a feature.

    This is useful for comparing multiple columns in the same DataFrame by a
    common feature.

    Note
    ----
      - When using qx the weight should be the exposure.
      - When using ae the weight should be the expected.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use.
    x_axis : str
        The name for the column to compare by.
    rates : list
        A list of rates to compare
    line_feature : str, optional (default=None)
        The name of the column to use for the line plot.
    weights : list, optional (default=None)
        A list of weights to weight the rates by.
    secondary : str, optional (default=None)
        The name of the column to have a secondary y-axis for.
    log_y : bool, optional (default=False)
        Whether to log the y-axis.
    x_bins : int, optional (default=None)
        The number of bins to use for the x-axis.
    display : bool, optional (default=True)
        Whether to display figure or now.
    **kwargs : dict
        Additional keyword arguments to pass to Plotly Express.

    Returns
    -------
    fig : Figure
        The chart

    """
    df = df.copy()
    # check if feature, secondary, line_feature is string and is in dataframe
    parameters = [x_axis, secondary, line_feature]
    for parameter in parameters:
        if parameter and not isinstance(parameter, str):
            raise ValueError(f"{parameter} should be a string.")
        if parameter and parameter not in df.columns:
            raise ValueError(
                f"Variable {parameter} is not in the DataFrame columns {df.columns}."
            )

    # check if weights list length matches the rates list
    if weights is not None and len(rates) != len(weights):
        logger.info(
            f"The weights list is {len(weights)} long and should "
            f"be {len(rates)} long. Using the first weight for all weights."
        )
        weights = [weights[0]] * len(rates)

    # group and calculate weighted means
    groupby_features = [x_axis]
    if line_feature:
        groupby_features.append(line_feature)
    if x_bins:
        logger.info(f"Binning feature: [{x_axis}] with {x_bins} bins")
        df[x_axis] = preprocessors.bin_feature(df[x_axis], x_bins)
    grouped_df = (
        df.groupby(groupby_features, observed=True)
        .apply(
            lambda x: pd.Series(
                {
                    **{
                        rate: helpers._weighted_mean(x[rate], weights=x[weight])
                        if weight
                        else x[rate].mean()
                        for rate, weight in zip(
                            rates,
                            [None] * len(rates) if weights is None else weights,
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

    # Add the bar chart for secondary variable first (background)
    fig.add_trace(
        go.Bar(
            x=grouped_df[x_axis],
            y=grouped_df[secondary],
            name=secondary,
            marker_color="rgba(135, 206, 250, 0.6)",
        ),
        secondary_y=True,
    )

    # Add the lines for rates (foreground)
    line_feature_values = grouped_df[line_feature].unique() if line_feature else [None]

    for rate in rates:
        for line_value in line_feature_values:
            if line_feature:
                df_subset = grouped_df[grouped_df[line_feature] == line_value]
                x_values, y_values = df_subset[x_axis], df_subset[rate]
                trace_name = f"{rate} - {line_value}"
            else:
                x_values, y_values = grouped_df[x_axis], grouped_df[rate]
                trace_name = rate

            # Add trace to the figure
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    name=trace_name,
                    mode="lines+markers",
                    **kwargs,
                ),
                secondary_y=False,
            )

    # Set plot layout
    yaxis_type = "-"
    y_title = "Rates"
    if log_y:
        yaxis_type = "log"
        y_title = "Log Rates"

    fig.update_layout(
        title_text=f"Comparison of {rates} by {x_axis}",
        yaxis_type=yaxis_type,
    )
    fig.update_xaxes(title_text=x_axis)
    fig.update_yaxes(title_text=y_title, secondary_y=False)
    fig.update_yaxes(title_text=secondary, secondary_y=True)

    # return the dataframe instead of figure
    if not display:
        fig = grouped_df

    return fig


def frequency(df, cols=1, features=None, sum_var=None):
    """
    Generate frequency plots.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use.
    cols : int, optional (default=1)
        The number of columns to use for the subplots.
    features : list, optional (default=None)
        The features to use for the plot. Default is to use all non-numeric features.
    sum_var : str, optional (default=None)
        The column name to use for the sum. Default is to use frequency.

    Returns
    -------
    fig : Figure
        The chart

    """
    # get the features for the plot
    if features is None:
        logger.info("No features provided, using all non-numeric features.")
        features = df.select_dtypes(exclude=[np.number]).columns.to_list()
    if sum_var is None:
        sum_var = "_count"
        df["_count"] = 1

    # creating the plot grid
    rows = len(features) // cols + (len(features) % cols > 0)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=features)

    # create frequency plot for each feature
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
        height=chart_height * rows,
        width=chart_width,
        showlegend=False,
        title_text=f"Frequency of variables using {sum_var}",
    )

    return fig


def pdp(
    model,
    df,
    x_axis,
    line_color=None,
    weight=None,
    secondary=None,
    mapping=None,
    x_bins=None,
    display=True,
):
    """
    Create a partial dependence plot (PDP) for the DataFrame.

    The 1-dimension partial dependency plot assumes that all of the features will
    use the weighted-average for that feature except the one that is being analyzed.
    That one will be predicted on. This then develops a relative risk as the only
    difference in prediction is the analyzed feature.

    The 2-dimension partial dependency plot is the same concept, however there will be
    two features where the values will be looped through.

    The pdp helps in understanding the relationship between variables, however there
    are only so many dimensions that can be modeled.

    reference: https://scikit-learn.org/stable/modules/partial_dependence.html

    Parameters
    ----------
    model : model
        The model to use that will be predicting
    df : pd.DataFrame
        The features in the model as well as the weights if applicable
    x_axis : str
        The feature to create the PDP for.
    line_color : str, optional (default=None)
        The feature to use for the line plot.
    weight : str, optional (default=None)
        The name of the column to use for the weights.
    secondary : str, optional (default=None)
        The name of the column to use for the secondary y-axis.
    mapping : dict, optional (default=None)
        A mapping dictionary that will lookup an encoded features original
        values.
    x_bins : int, optional (default=None)
        The number of bins to use for the x-axis.
    display : bool, optional (default=True)
        Whether to display figure or now.

    Returns
    -------
    fig : Figure
        The chart

    """
    df = df.copy()
    # make sure model has prediction function
    if not hasattr(model, "predict"):
        raise ValueError(f"Model: [{model}] does not have a predict function.")
    logger.info(f"Model: [{type(model).__name__}] for partial dependence plot.")

    # initialize variables
    x_axis_type = "passthrough"
    line_color_type = "passthrough"

    grouped_features = [x_axis]
    colorscale = px.colors.qualitative.G10
    weights = None
    if weight:
        weights = df[weight]
        logger.info(f"Weights: [{weight}]")

    # get the feature names from the model
    model_features = None
    feature_attrs = ["feature_names_in_", "params"]
    for attr in feature_attrs:
        try:
            if attr == "params":
                model_features = list(getattr(model, attr).keys())
                df["const"] = 1
            else:
                model_features = list(getattr(model, attr))
            if model_features:
                break
        except AttributeError:
            continue
    if not model_features:
        raise ValueError("Model does not have feature names.")
    X = df[model_features]

    # check df is not empty
    if df.empty:
        raise ValueError("DataFrame is empty.")

    # x_axis processing
    logger.info(f"x_axis: [{x_axis}] type: [{x_axis_type}]")
    if mapping and x_axis in mapping:
        x_axis_type = mapping[x_axis]["type"]
        if x_axis_type == "ohe":
            x_axis_cols = [col for col in X.columns if col.startswith(x_axis + "_")]
            x_axis_values = [col[len(x_axis + "_") :] for col in x_axis_cols]
        else:
            x_axis_values = list(X[x_axis].unique())
    else:
        x_axis_values = np.linspace(X[x_axis].min(), X[x_axis].max(), 100)

    # line_color processing
    if line_color:
        line_color_type = mapping[line_color]["type"]
        logger.info(f"Line feature: [{line_color}] type: [{line_color_type}]")
        line_color_values = X[line_color].unique()
    else:
        line_color = "Overall"
        df[line_color] = "Overall"
        line_color_values = df[line_color].unique()
    grouped_features.append(line_color)

    # secondary processing
    if secondary:
        secondary_df = (
            df.groupby(grouped_features, observed=True)[secondary]
            .sum()
            .reset_index()
            .sort_values(by=grouped_features)
        )

    # average value of features
    X_mean = X.apply(lambda x: helpers._weighted_mean(x, weights=weights))
    if isinstance(X_mean.dtype, pd.SparseDtype):
        X_mean = X_mean.sparse.to_dense()

    # calculate predictions of feature by looping through the feature values
    # and using the average of the other features
    preds = []
    for line_value in line_color_values:
        for value in x_axis_values:
            X_temp = X_mean.copy()

            # ohe zeros out the encoded columns and steps through the values
            if x_axis_type == "ohe":
                for col in x_axis_cols:
                    X_temp[col] = 0
                X_temp[x_axis + "_" + value] = 1
            else:
                X_temp[x_axis] = value

            # creating line_color values
            if line_color and line_color != "Overall":
                X_temp[line_color] = line_value

            # change series to dataframe
            if isinstance(X_temp, pd.Series):
                X_temp = X_temp.to_frame().T

            # predict the value
            pred = model.predict(X_temp)[0]
            pred_dict = {
                x_axis: value,
                line_color: line_value,
                "pred": pred,
            }
            preds.append(pred_dict)

    plot_df = pd.DataFrame(preds)

    # the mean prediction should be average and not weighted so that the values
    # are relative to eachother and not to the weights.
    mean_pred = np.mean(plot_df["pred"])
    plot_df["%_diff"] = (plot_df["pred"] - mean_pred) / mean_pred + 1
    if secondary:
        plot_df = plot_df.merge(secondary_df, on=grouped_features, how="left")

    # use mapping to get the original x_axis values
    if mapping and x_axis in mapping and x_axis_type != "ohe":
        reversed_mapping = {v: k for k, v in mapping[x_axis]["values"].items()}
        plot_df[x_axis] = plot_df[x_axis].map(reversed_mapping)
    if mapping and line_color and line_color in mapping and line_color != "ohe":
        reversed_mapping = {v: k for k, v in mapping[line_color]["values"].items()}
        plot_df[line_color] = plot_df[line_color].map(reversed_mapping)

    plot_df = plot_df.sort_values(by=grouped_features)

    # bin the feature if x_bins is provided
    if x_bins:
        logger.info(f"Binning feature: [{x_axis}] with {x_bins} bins")
        plot_df[x_axis] = preprocessors.bin_feature(plot_df[x_axis], x_bins)
        if secondary:
            plot_df = (
                plot_df.groupby(grouped_features, observed=True)
                .agg({"%_diff": "mean", secondary: "sum"})
                .reset_index()
            )
        else:
            plot_df = (
                plot_df.groupby(grouped_features, observed=True).mean().reset_index()
            )

    # create the plots
    rows = 2 if secondary else 1
    fig = make_subplots(rows=rows, cols=1)

    # add the line plot
    for index, line_color_value in enumerate(plot_df[line_color].unique()):
        df_subset = plot_df[plot_df[line_color] == line_color_value]
        fig.add_trace(
            go.Scatter(
                x=df_subset[x_axis],
                y=df_subset["%_diff"],
                mode="lines",
                name=line_color_value,
                line={"color": colorscale[index]},
            ),
            row=1,
            col=1,
        )
    fig.update_layout(
        title="Partial Dependency Plot",
        yaxis_title="%_diff",
        yaxis_tickformat=".1%",
        legend_title=line_color if line_color else "overall",
        height=400 * rows,
        width=chart_width,
    )

    # add in seconary plot
    if secondary:
        logger.info(f"Adding secondary to chart: [{secondary}]")

        for index, line_color_value in enumerate(plot_df[line_color].unique()):
            df_subset = plot_df[plot_df[line_color] == line_color_value]
            fig.add_trace(
                go.Bar(
                    x=df_subset[x_axis],
                    y=df_subset[secondary],
                    name=line_color_value,
                    marker={"color": colorscale[index]},
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            yaxis2_title=secondary,
            yaxis2_tickformat=",",
        )

    if not display:
        fig = plot_df

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
    fig.update_layout(
        height=chart_height * num_rows, width=chart_width, title_text="Scatter Plots"
    )

    return fig


def target(
    df, target, features=None, cols=3, numerator=None, denominator=None, normalize=None
):
    """
    Create multiplot showing variable relationship with target.

    If choosing a column for the target, the target will be the mean of the column for
    the feature. If choosing "ratio" or "risk" then the numerator and denominator
    columns are required and the target will be the ratio or risk of the two columns.

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
    normalize : list, optional
        The columns to normalize.

    Returns
    -------
    fig : Figure
        The chart

    """
    # default parameters
    if features is None:
        features = df.columns
    if normalize is None:
        normalize = []

    # validations
    if target not in [*list(df.columns), "ratio", "risk"]:
        raise ValueError(
            f"Target '{target}' needs to be in DataFrame columns, 'ratio', or 'risk'"
        )
    if not set(features).issubset(df.columns):
        missing_features = set(features) - set(df.columns)
        raise ValueError(f"Features {missing_features} not in DataFrame columns.")
    if not set(normalize).issubset(df.columns):
        missing_features = set(normalize) - set(df.columns)
        raise ValueError(f"Normalize {missing_features} not in DataFrame columns.")
    if target in ["ratio", "risk"] and (numerator is None or denominator is None):
        raise ValueError("Numerator/Denominator is required for ratio or risk target.")

    # normalize if requested
    if normalize:
        df = experience.normalize(
            df, features=normalize, numerator=numerator, denominator=denominator
        )
        numerator = f"{numerator}_norm"

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
            ).sort_values(by=feature)
            grouped_data[target] = grouped_data[numerator] / grouped_data[denominator]
        elif target == "risk":
            grouped_data = (
                df.groupby(feature, observed=True)[[numerator, denominator]]
                .sum()
                .reset_index()
            ).sort_values(by=feature)
            total_target = (
                grouped_data[numerator].sum() / grouped_data[denominator].sum()
            )
            grouped_data[target] = (
                grouped_data[numerator] / grouped_data[denominator]
            ) / total_target
        else:
            grouped_data = (
                df.groupby(feature, observed=True)[target].mean().reset_index()
            ).sort_values(by=feature)
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
    fig.update_layout(
        height=chart_height * num_rows, width=chart_width, title_text="Target Plots"
    )

    return fig
