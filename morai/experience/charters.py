"""Collection of visualization tools."""

import itertools
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from joblib import Parallel, cpu_count, delayed
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

from morai.experience import experience
from morai.forecast import models, preprocessors
from morai.utils import custom_logger, helpers

logger = custom_logger.setup_logging(__name__)

# default chart height and width
chart_height = 400
chart_width = 1000


def chart(
    df: Union[pd.DataFrame, pl.LazyFrame],
    x_axis: str,
    y_axis: Optional[str] = None,
    color: Optional[str] = None,
    type: str = "line",
    numerator: Optional[str] = None,
    denominator: Optional[str] = None,
    title: Optional[str] = None,
    y_sort: bool = False,
    x_bins: Optional[int] = None,
    y_log: bool = False,
    add_line: bool = False,
    agg: str = "sum",
    display: bool = True,
    **kwargs: Any,
) -> Union[go.Figure, Union[pd.DataFrame, pl.LazyFrame]]:
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
    df : Union[pd.DataFrame, pl.LazyFrame]
        The dataframe or LazyFrame to chart.
    x_axis : str
        The column name to use for the x-axis.
    y_axis : str
        The column name to use for the y-axis.
        ["ratio", "risk"] are special values that will calculate the ratio
        of two columns.
    color : str, optional (default=None)
        The column name to use for the color.
    type : str, optional (default="line")
        The type of chart to create. Options are "line", "heatmap", "bar", or "area".
    numerator : str, optional (default=None)
        Only used when y_axis is "ratio" or "risk".
        The column name to use for the numerator values.
    denominator : str, optional (default=None)
        Only used when y_axis is "ratio" or "risk".
        The column name to use for the denominator values.
    title : str
        The title of the chart.
    y_sort : bool, optional (default=False)
        Sort the records by the y-axis descending.
    x_bins : int, optional (default=None)
        The number of bins to use for the x-axis.
    y_log : bool, optional (default=False)
        Whether to log the y-axis.
    add_line : bool, optional (default=False)
        Whether to add a line to the chart at y-axis of 1.
    agg : str, optional (default="sum")
        The aggregation to use for the y-axis.
    display : bool, optional (default=True)
        Whether to display figure or not.
    **kwargs : dict
        Additional keyword arguments to pass to Plotly Express.

    Returns
    -------
    fig : Figure or DataFrame/LazyFrame
        The chart or grouped data if display=False

    """
    # lazyframe check
    is_lazy = isinstance(df, pl.LazyFrame)

    # initialize variables
    if not is_lazy:
        df = df.copy()
    use_num_and_den = True if numerator and denominator else False

    # heatmap sum values are color rather than y_axis
    if type in ("heatmap", "contour"):
        if not color:
            raise ValueError("Color parameter is required for heatmap/contour.")
        _y_axis = y_axis
        _color = color
        color = _y_axis
        y_axis = _color

    yaxis_type = "-"
    if y_log:
        yaxis_type = "log"

    # getting the columns to sum by
    # columns will be y_axis unless y_axis is "ratio" or "risk"
    check_cols = [x_axis, y_axis, color, numerator, denominator]
    if use_num_and_den and y_axis in ["ratio", "risk"]:
        logger.info(f"Calculating {y_axis} using [{numerator}] and [{denominator}]")
        agg_cols = [numerator, denominator]
        check_cols.remove(y_axis)
    else:
        agg_cols = y_axis

    # check if missing columns
    if is_lazy:
        schema = df.collect_schema()
        columns = list(schema.keys())
        missing_columns = [
            col for col in check_cols if col is not None and col not in columns
        ]
    else:
        missing_columns = [
            col for col in check_cols if col is not None and col not in df.columns
        ]
    if missing_columns:
        raise ValueError(
            f"The following column(s) are not in the "
            f"DataFrame: {', '.join(missing_columns)}"
        )

    # groupby by the x_axis and color
    groupby_cols = [x_axis]
    if color:
        groupby_cols.append(color)

    # group data
    if is_lazy:
        if x_bins:
            logger.info(f"Binning feature: [{x_axis}] with {x_bins} bins")
            df = preprocessors.lazy_bin_feature(df, x_axis, x_bins, inplace=True)

        grouped_data = preprocessors.lazy_groupby(df, groupby_cols, agg_cols, agg)
        grouped_data = grouped_data.collect().to_pandas()

    else:  # pandas
        if x_bins:
            logger.info(f"Binning feature: [{x_axis}] with {x_bins} bins")
            df[x_axis] = preprocessors.bin_feature(df[x_axis], x_bins)

        grouped_data = (
            df.groupby(groupby_cols, observed=True)[agg_cols].agg(agg).reset_index()
        )
    grouped_data = grouped_data.sort_values(
        groupby_cols,
        key=lambda x: x.astype(str) if pd.api.types.is_categorical_dtype(x) else x,
    )

    # calculate ratios if needed
    if use_num_and_den and y_axis == "ratio":
        grouped_data[y_axis] = grouped_data[numerator] / grouped_data[denominator]
    elif use_num_and_den and y_axis == "risk":
        average_y_axis = grouped_data[numerator].sum() / grouped_data[denominator].sum()
        grouped_data[y_axis] = (
            grouped_data[numerator] / grouped_data[denominator]
        ) / average_y_axis

    # sort
    if y_sort:
        grouped_data = grouped_data.sort_values(by=y_axis, ascending=False)

    # return data if not display
    if not display:
        return grouped_data

    # Selecting the plot type based on the 'chart_type' parameter
    if type == "line":
        if not title:
            title = f"'{y_axis}' by '{x_axis}' and '{color}'"
        fig = px.line(
            grouped_data,
            x=x_axis,
            y=y_axis,
            color=color,
            title=title,
            **kwargs,
        )
        if add_line:
            # using scatter to add to legend
            fig.add_scatter(
                x=[grouped_data[x_axis].min(), grouped_data[x_axis].max()],
                y=[1, 1],
                mode="lines",
                line={"dash": "dot", "color": "grey"},
                name="y=1",
            )
    elif type == "bar":
        if not title:
            title = f"'{y_axis}' by '{x_axis}' and '{color}'"
        fig = px.bar(
            grouped_data,
            x=x_axis,
            y=y_axis,
            color=color,
            title=title,
            **kwargs,
        )
    elif type == "area":
        if not title:
            title = f"'{y_axis}' by '{x_axis}' and '{color}'"
        fig = px.area(
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
            title = f"Heatmap of '{_color}' by '{x_axis}' and '{_y_axis}'"
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=grouped_data.values,
                    x=grouped_data.columns,
                    y=grouped_data.index,
                    **kwargs,
                )
            ]
        )
        fig.update_layout(title=title, xaxis_title=x_axis, yaxis_title=_y_axis)
    elif type == "contour":
        grouped_data = grouped_data.pivot(index=_y_axis, columns=x_axis, values=_color)
        if not title:
            title = f"Contour of '{_color}' by '{x_axis}' and '{_y_axis}'"
        fig = go.Figure(
            data=[
                go.Contour(
                    z=grouped_data.values,
                    x=grouped_data.columns,
                    y=grouped_data.index,
                    **kwargs,
                )
            ]
        )
        fig.update_layout(
            title=title,
            xaxis_title=x_axis,
            yaxis_title=_y_axis,
        )
    else:
        raise ValueError(
            "Unsupported type. Use 'line', 'bar', 'heatmap', or 'contour'."
        )

    fig.update_layout(
        yaxis_type=yaxis_type,
        height=chart_height,
    )

    return fig


def compare_rates(
    df: Union[pd.DataFrame, pl.LazyFrame],
    x_axis: str,
    rates: List[str],
    line_feature: Optional[str] = None,
    weights: Optional[List[str]] = None,
    secondary: Optional[str] = None,
    y_log: bool = False,
    x_bins: Optional[int] = None,
    display: bool = True,
    **kwargs: Any,
) -> Union[go.Figure, Union[pd.DataFrame, pl.LazyFrame]]:
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
    df : Union[pd.DataFrame, pl.LazyFrame]
        The DataFrame or LazyFrame to use.
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
    y_log : bool, optional (default=False)
        Whether to log the y-axis.
    x_bins : int, optional (default=None)
        The number of bins to use for the x-axis.
    display : bool, optional (default=True)
        Whether to display figure or now.
    **kwargs : dict
        Additional keyword arguments to pass to Plotly Express.

    Returns
    -------
    fig : Figure or DataFrame/LazyFrame
        The chart or grouped data if display=False

    """
    # check if lazy
    is_lazy = isinstance(df, pl.LazyFrame)

    if is_lazy:
        if df.collect().height == 0:
            logger.warning("DataFrame is empty.")
            return go.Figure()
        schema = df.collect_schema()
        columns = list(schema.keys())
    else:
        df = df.copy()
        if df.empty:
            logger.warning("DataFrame is empty.")
            return go.Figure()
        columns = df.columns

    # check parameters
    parameters = [x_axis, secondary, line_feature]
    for parameter in parameters:
        if parameter and not isinstance(parameter, str):
            raise ValueError(f"{parameter} should be a string.")
        if parameter and parameter not in columns:
            raise ValueError(
                f"Variable {parameter} is not in the DataFrame columns {columns}."
            )

    # check weights
    if weights is not None and len(rates) != len(weights):
        logger.info(
            f"The weights list is {len(weights)} long and should "
            f"be {len(rates)} long. Using the first weight for all weights."
        )
        weights = [weights[0]] * len(rates)

    groupby_features = [x_axis]
    if line_feature:
        groupby_features.append(line_feature)

    # create grouped data
    if is_lazy:
        if x_bins:
            logger.info(f"Binning feature: [{x_axis}] with {x_bins} bins")
            df = preprocessors.lazy_bin_feature(df, x_axis, x_bins, inplace=True)
        agg_exprs = []
        for rate, weight in zip(
            rates, [None] * len(rates) if weights is None else weights, strict=False
        ):
            if weight:
                agg_exprs.append(
                    (
                        (pl.col(rate) * pl.col(weight)).sum() / pl.col(weight).sum()
                    ).alias(rate)
                )
            else:
                agg_exprs.append(pl.mean(pl.col(rate)).alias(rate))
        if secondary:
            agg_exprs.append(pl.sum(secondary).alias(secondary))

        grouped_data = (
            df.group_by(groupby_features).agg(agg_exprs).collect().to_pandas()
        )
    else:  # pandas
        if x_bins:
            logger.info(f"Binning feature: [{x_axis}] with {x_bins} bins")
            df[x_axis] = preprocessors.bin_feature(df[x_axis], x_bins)
        grouped_data = (
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
                                strict=False,
                            )
                        },
                        **({secondary: x[secondary].sum()} if secondary else {}),
                    }
                )
            )
            .reset_index()
        )
    grouped_data = grouped_data.sort_values(
        groupby_features,
        key=lambda x: x.astype(str) if pd.api.types.is_categorical_dtype(x) else x,
    )

    # return data if not display
    if not display:
        return grouped_data

    # create figures
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if secondary:
        fig.add_trace(
            go.Bar(
                x=grouped_data[x_axis],
                y=grouped_data[secondary],
                name=secondary,
                marker_color="rgba(135, 206, 250, 0.6)",
            ),
            secondary_y=True,
        )

    # add lines
    line_feature_values = (
        grouped_data[line_feature].unique() if line_feature else [None]
    )

    for rate in rates:
        for line_value in line_feature_values:
            if line_feature:
                df_subset = grouped_data[grouped_data[line_feature] == line_value]
                x_values, y_values = df_subset[x_axis], df_subset[rate]
                trace_name = f"{rate} - {line_value}"
            else:
                x_values, y_values = grouped_data[x_axis], grouped_data[rate]
                trace_name = rate

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

    # plot layout
    yaxis_type = "-"
    y_title = "Rates"
    if y_log:
        yaxis_type = "log"
        y_title = "Log Rates"

    fig.update_layout(
        title_text=f"Comparison of '{rates}' by '{x_axis}'",
        yaxis_type=yaxis_type,
        yaxis_title=y_title,
    )

    return fig


def frequency(
    df: Union[pd.DataFrame, pl.LazyFrame],
    cols: int = 1,
    features: Optional[List[str]] = None,
    sum_var: Optional[str] = None,
) -> go.Figure:
    """
    Generate frequency plots.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.LazyFrame]
        The DataFrame or LazyFrame to use.
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
    is_lazy = isinstance(df, pl.LazyFrame)

    # get the non-numeric features for the plot
    if features is None:
        logger.info("No features provided, using all non-numeric features.")
        if is_lazy:
            schema = df.collect_schema()
            features = [
                col
                for col, dtype in schema.items()
                if schema[col] not in pl.datatypes.group.NUMERIC_DTYPES
            ]
        else:
            features = df.select_dtypes(exclude=[np.number]).columns.to_list()

    if sum_var is None:
        sum_var = "_count"
        if is_lazy:
            df = df.with_columns(pl.lit(1).alias("_count"))
        else:
            df["_count"] = 1

    # creating the plot grid
    rows = len(features) // cols + (len(features) % cols > 0)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=features)

    # create frequency plot for each feature
    for i, col in enumerate(features, start=1):
        if is_lazy:
            frequency_lazy = df.group_by(col).agg(pl.sum(sum_var).alias(sum_var))
            frequency_lazy = frequency_lazy.sort(col)
            frequency = frequency_lazy.collect().to_pandas()
        else:
            frequency = df.groupby(col, observed=True)[sum_var].sum().reset_index()
            frequency.columns = [col, sum_var]
            frequency = frequency.sort_values(
                by=col,
                ascending=True,
                key=lambda x: x.astype(str)
                if pd.api.types.is_categorical_dtype(x)
                else x,
            )

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
        title_text=f"Frequency of variables using '{sum_var}'",
    )

    return fig


def pdp(
    model: Any,
    df: pd.DataFrame,
    x_axis: str,
    line_color: Optional[str] = None,
    weight: Optional[str] = None,
    secondary: Optional[str] = None,
    mapping: Optional[Dict[str, Dict[str, Union[str, Dict[str, str]]]]] = None,
    x_bins: Optional[int] = None,
    quick: bool = False,
    n_jobs: Optional[int] = None,
    display: bool = True,
) -> Union[go.Figure, pd.DataFrame]:
    """
    Create a partial dependence plot (PDP) for the DataFrame.

    The 1-dimension partial dependency plot assumes will loop through the values of
    the feature and average the predictions.
    This then develops a relative risk as the only difference in prediction is
    the analyzed feature.

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
    quick : bool, optional (default=False)
        Whether to use a quicker method for pdp, however the results may not be as
        accurate.
    n_jobs : int, optional
        Number of parallel jobs to run. If None, the computation is sequential.
        n_jobs=-1 means using all processors.
    display : bool, optional (default=True)
        Whether to display figure or now.

    Returns
    -------
    fig : Figure
        The chart

    """
    df = df.copy()
    # make sure model has prediction function
    models.ModelWrapper(model).check_predict()
    logger.info(f"Model: [{type(model).__name__}] for partial dependence plot.")

    # initialize variables
    x_axis_type = "passthrough"
    line_color_type = "passthrough"
    x_axis_cols = None

    grouped_features = [x_axis]
    weights = None
    if weight:
        weights = df[weight]
        logger.info(f"Weights: [{weight}]")

    # get the feature names from the model to create X
    model_features = None
    model_features = models.ModelWrapper(model).get_features()
    logger.debug(f"Model features for pdp: {model_features}")

    # check df is not empty
    if df[model_features].empty:
        raise ValueError("DataFrame is empty.")

    # x_axis processing
    if mapping and x_axis in mapping:
        x_axis_type = mapping[x_axis]["type"]
        if x_axis_type == "ohe":
            x_axis_cols = list(mapping[x_axis]["values"].values())
            x_axis_values = list(mapping[x_axis]["values"].keys())
        else:
            x_axis_values = list(df[x_axis].unique())
    elif x_axis in df.select_dtypes(exclude=[np.number]).columns:
        x_axis_values = list(df[x_axis].unique())
    else:
        # linspace is creating float falues
        if pd.api.types.is_integer_dtype(df[x_axis].dtype):
            df[x_axis] = df[x_axis].astype(float)
        x_axis_values = np.linspace(df[x_axis].min(), df[x_axis].max(), 100)
    logger.info(f"x_axis: [{x_axis}] type: [{x_axis_type}]")
    X = df[model_features]

    # line_color processing
    if line_color:
        line_color_type = mapping[line_color]["type"]
        if line_color_type == "ohe":
            raise ValueError("One hot encoding not supported for line_color.")
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
            .sort_values(
                by=grouped_features,
                key=lambda x: x.astype(str)
                if pd.api.types.is_categorical_dtype(x)
                else x,
            )
        )

    # quick method for pdp
    if quick:
        logger.warning("Using a quick method for pdp. Results may not be as accurate.")
        # check which columns are not numeric
        str_cols = X.select_dtypes(exclude=[np.number]).columns.to_list()
        if str_cols:
            logger.warning(
                f"quick method only works with numeric columns.\n"
                f"string columns: {str_cols}"
            )
            quick = False
        else:
            X = (
                X.apply(lambda x: helpers._weighted_mean(x, weights=weights))
                .to_frame()
                .T
            )

    # get the amount of iterations needed by getting combo of x_axis and line_color
    logger.info(f"Creating {len(x_axis_values) * len(line_color_values)} predictions.")

    # calculate predictions of feature by looping through the feature values
    # and using the average of the other features
    if n_jobs:
        if n_jobs == -1:
            n_jobs = cpu_count()
        logger.info(f"Running '{n_jobs}' cores for parallel processing.")
        verbose = 0
        if custom_logger.get_log_level() == "DEBUG":
            verbose = 10
        preds = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_pdp_make_prediction)(
                model,
                X,
                x_axis,
                x_axis_type,
                x_axis_cols,
                value,
                line_color,
                line_value,
                quick,
                weights,
            )
            for line_value, value in tqdm(
                list(itertools.product(line_color_values, x_axis_values)),
                desc="Processing",
                unit="combo",
            )
        )
    else:
        preds = []
        for line_value, value in tqdm(
            list(itertools.product(line_color_values, x_axis_values)),
            desc="Processing",
            unit="combo",
        ):
            pred = _pdp_make_prediction(
                model,
                X,
                x_axis,
                x_axis_type,
                x_axis_cols,
                value,
                line_color,
                line_value,
                quick,
                weights,
            )
            preds.append(pred)

    pdp_df = pd.DataFrame(preds)

    # the mean prediction should be average and not weighted so that the values
    # are relative to eachother and not to the weights.
    mean_pred = np.mean(pdp_df["pred"])
    pdp_df["%_diff"] = (pdp_df["pred"] - mean_pred) / mean_pred + 1
    if secondary:
        pdp_df = pdp_df.merge(secondary_df, on=grouped_features, how="left")

    # use mapping to get the original x_axis values
    if mapping and x_axis in mapping and x_axis_type != "ohe":
        pdp_df[x_axis] = preprocessors.remap_values(pdp_df[x_axis], mapping)
    if mapping and line_color and line_color in mapping and line_color != "ohe":
        pdp_df[line_color] = preprocessors.remap_values(pdp_df[line_color], mapping)

    pdp_df = pdp_df.sort_values(
        by=grouped_features,
        key=lambda x: x.astype(str) if pd.api.types.is_categorical_dtype(x) else x,
    )

    # bin the feature if x_bins is provided
    # note: can't bin prior to prediction, because the prediction is based on the
    # model that used the original feature values.
    if x_bins:
        logger.info(f"Binning feature: [{x_axis}] with {x_bins} bins")
        pdp_df[x_axis] = preprocessors.bin_feature(pdp_df[x_axis], x_bins)
        if secondary:
            pdp_df = (
                pdp_df.groupby(grouped_features, observed=True)
                .agg({"%_diff": "mean", secondary: "sum"})
                .reset_index()
            )
        else:
            pdp_df = (
                pdp_df.groupby(grouped_features, observed=True).mean().reset_index()
            )

    # create the plots
    colorscale = px.colors.qualitative.Light24
    num_colors = len(colorscale)
    rows = 2 if secondary else 1
    fig = make_subplots(rows=rows, cols=1)

    # add the line plot
    for index, line_color_value in enumerate(pdp_df[line_color].unique()):
        color_index = index % num_colors
        df_subset = pdp_df[pdp_df[line_color] == line_color_value]
        fig.add_trace(
            go.Scatter(
                x=df_subset[x_axis],
                y=df_subset["%_diff"],
                mode="lines",
                name=line_color_value,
                line={"color": colorscale[color_index]},
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

        for index, line_color_value in enumerate(pdp_df[line_color].unique()):
            color_index = index % num_colors
            df_subset = pdp_df[pdp_df[line_color] == line_color_value]
            fig.add_trace(
                go.Bar(
                    x=df_subset[x_axis],
                    y=df_subset[secondary],
                    name=line_color_value,
                    marker={"color": colorscale[color_index]},
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            yaxis2_title=secondary,
            yaxis2_tickformat=",",
        )

    if not display:
        fig = pdp_df

    return fig


def scatter(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    sample_nbr: float = 100,
    cols: int = 3,
) -> go.Figure:
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


def matrix(
    df: pd.DataFrame, threshold: float = 0.5, title: str = "Matrix Heatmap"
) -> go.Figure:
    """
    Create a heatmap of a matrix dataframe.

    Used to show pairwise correlation of features.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use that is a matrix.
    threshold : float, optional
        The threshold to use for the heatmap.
    title : str, optional
        The title of the chart.

    Returns
    -------
    fig : Figure
        The chart

    """
    # mask the upper triangle
    mask = np.triu(np.ones_like(df, dtype=bool))
    df = df.mask(mask)

    # create the plot
    fig = px.imshow(
        df,
        labels={"x": "features", "y": "features", "color": "value"},
        color_continuous_scale="Blues",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )

    fig.update_layout(
        title=title,
        plot_bgcolor="gray",
        xaxis={"showgrid": False},
        yaxis={"showgrid": False},
    )

    # add annotations for significant correlations
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if not mask[i, j]:
                if df.iloc[i, j] < -threshold or df.iloc[i, j] > threshold:
                    fig.add_annotation(
                        x=df.columns[j],
                        y=df.index[i],
                        text=f"{df.iloc[i, j]:.2f}",
                        showarrow=False,
                        font={"color": "red"},
                    )
                else:
                    fig.add_annotation(
                        x=df.columns[j],
                        y=df.index[i],
                        text=f"{df.iloc[i, j]:.2f}",
                        showarrow=False,
                    )

    return fig


def target(
    df: Union[pd.DataFrame, pl.LazyFrame],
    target: Union[List[str], str],
    features: Optional[List[str]] = None,
    cols: int = 3,
    numerator: Optional[Union[List[str], str]] = None,
    denominator: Optional[Union[List[str], str]] = None,
    weights: Optional[Union[List[str], str]] = None,
    normalize: Optional[List[str]] = None,
    add_line: bool = False,
    pairwise: bool = False,
) -> go.Figure:
    """
    Create multiplot showing variable relationship with target.

    If choosing a column for the target, the target will be the mean of the column for
    the feature. If choosing "ratio" or "risk" then the numerator and denominator
    columns are required and the target will be the ratio or risk of the two columns.

    Parameters
    ----------
    df : pd.DataFrame or pl.LazyFrame
        The DataFrame or LazyFrame to use.
    target : list or str
        The target variable.
    features : list, optional
        The features to use for the plot. Default is to use all features.
    cols : int, optional
        The number of columns to use for the subplots.
    numerator : list, optional
        The column name to use for the numerator values.
    denominator : list, optional
        The column name to use for the denominator values.
    weights : list, optional
        The column name to use for the weights values.
    normalize : list, optional
        The columns to normalize.
    add_line : bool, optional
        Whether to add a line to the chart at y-axis of 1.
    pairwise : bool, optional
        Whether to create a pairwise plot of the features.

    Returns
    -------
    fig : Figure
        The chart

    """
    # check if lazy
    is_lazy = isinstance(df, pl.LazyFrame)

    # default parameters
    if features is None:
        if is_lazy:
            schema = df.collect_schema()
            features = list(schema.keys())
        else:  # pandas
            features = df.columns

    if normalize is None:
        normalize = []
    if denominator is None:
        denominator = []
    if numerator is None:
        numerator = []
    if weights is None:
        weights = []

    features = [*features, "_aggregate"]

    # add _aggregate column
    if is_lazy:
        df = df.with_columns(pl.lit(1).alias("_aggregate"))
    else:  # pandas
        df.loc[:, "_aggregate"] = 1

    if isinstance(target, str):
        target = [target]
    if isinstance(weights, str):
        weights = [weights]

    # validations
    # make target, weights, numerator same length
    if len(denominator) != 0 and len(target) != len(denominator):
        logger.debug(
            f"Target is updated to match denominator length: {len(denominator)}"
        )
        target = [target[0]] * len(denominator)
    if len(weights) != 0 and len(weights) != len(target):
        weights = [weights[0]] * len(target)
    if len(denominator) != 0 and len(numerator) != len(denominator):
        logger.debug(
            f"Numerator is updated to match denominator length: {len(denominator)}"
        )
        numerator = [numerator[0]] * len(denominator)

    # ensure all features are in df
    if is_lazy:
        schema = df.collect_schema()
        df_columns = list(schema.keys())
    else:  # pandas
        df_columns = df.columns

    if not set(features).issubset(df_columns):
        missing_features = set(features) - set(df_columns)
        raise ValueError(f"Features {missing_features} not in DataFrame columns.")
    if not set(normalize).issubset(df_columns):
        missing_features = set(normalize) - set(df_columns)
        raise ValueError(f"Normalize {missing_features} not in DataFrame columns.")

    # check that the right parameters are set for the function
    for idx, _ in enumerate(target):
        if target[idx] not in [*list(df_columns), "ratio", "risk"]:
            raise ValueError(
                f"Target '{target[idx]}' needs to be in DataFrame columns, "
                f"'ratio', or 'risk'"
            )
        if target[idx] in ["ratio", "risk"] and (
            numerator[idx] is None or denominator[idx] is None
        ):
            raise ValueError(
                "Numerator/Denominator is required for ratio or risk target."
            )
        if target[idx] not in ["ratio", "risk"] and (
            len(numerator) + len(denominator) > 0
        ):
            logger.warning(
                "Parameters 'numerator' and 'denominator' are ignored if target is "
                "not 'ratio' or 'risk'."
            )

    # normalize if requested
    if normalize:
        for idx, _ in enumerate(target):
            df = experience.normalize(
                df,
                features=normalize,
                numerator=numerator[idx],
                denominator=denominator[idx],
                suffix=f"{idx}",
            )
            numerator[idx] = f"{numerator[idx]}_norm_{idx}"

    # number of rows for the subplot grid
    if pairwise:
        features.remove("_aggregate")
        feature_pairs = [list(pair) for pair in itertools.combinations(features, 2)]
        num_plots = len(feature_pairs)
        num_rows = math.ceil(num_plots / cols)
        plot_features = feature_pairs
        title = f"Pairwise Plots using '{target}'"
    else:
        num_plots = len(features)
        num_rows = math.ceil(num_plots / cols)
        plot_features = [[feature] for feature in features]
        title = f"Target Plots using '{target}'"

    # create a subplot grid
    logger.info(f"Creating '{num_plots}' target plots.")
    fig = make_subplots(
        rows=num_rows,
        cols=cols,
        subplot_titles=[", ".join(pair) for pair in plot_features],
    )
    legend_added = set()
    # color scale to cycle through
    # https://plotly.com/python/discrete-color/
    color_scale = px.colors.qualitative.D3

    # create each plot for the feature
    for i, plot_feature in enumerate(plot_features, 1):
        row = (i - 1) // cols + 1
        col = (i - 1) % cols + 1

        # create line for targets
        # ratio or risk
        for idx, _ in enumerate(target):
            if target[idx] in ["ratio", "risk"]:
                if is_lazy:
                    grouped_data = (
                        df.group_by(plot_feature)
                        .agg(
                            [
                                pl.sum(numerator[idx]).alias(numerator[idx]),
                                pl.sum(denominator[idx]).alias(denominator[idx]),
                            ]
                        )
                        .collect()
                        .to_pandas()
                    )
                    if target[idx] == "ratio":
                        grouped_data[target[idx]] = (
                            grouped_data[numerator[idx]]
                            / grouped_data[denominator[idx]]
                        )
                    elif target[idx] == "risk":
                        num_sum = (
                            df.select(pl.col(numerator[idx]).sum()).collect().item()
                        )
                        denom_sum = (
                            df.select(pl.col(denominator[idx]).sum()).collect().item()
                        )
                        grouped_data[target[idx]] = (
                            grouped_data[numerator[idx]]
                            / grouped_data[denominator[idx]]
                        ) / (num_sum / denom_sum)
                else:  # pandas
                    grouped_data = (
                        df.groupby(plot_feature, observed=True)[
                            [numerator[idx], denominator[idx]]
                        ]
                        .sum()
                        .reset_index()
                    )
                    if target[idx] == "ratio":
                        grouped_data[target[idx]] = (
                            grouped_data[numerator[idx]]
                            / grouped_data[denominator[idx]]
                        )
                    elif target[idx] == "risk":
                        grouped_data[target[idx]] = (
                            grouped_data[numerator[idx]]
                            / grouped_data[denominator[idx]]
                        ) / (df[numerator[idx]].sum() / df[denominator[idx]].sum())

            # weighted average
            elif weights:
                if is_lazy:
                    grouped_data = (
                        df.select([*plot_feature, target[idx], weights[idx]])
                        .groupby(plot_feature)
                        .agg(
                            [
                                (pl.col(target[idx]) * pl.col(weights[idx]))
                                .sum()
                                .alias("weighted_sum"),
                                pl.col(weights[idx]).sum().alias("weight_sum"),
                            ]
                        )
                        .with_columns(
                            (pl.col("weighted_sum") / pl.col("weight_sum")).alias(
                                target[idx]
                            )
                        )
                        .select([*plot_feature, target[idx]])
                        .collect()
                        .to_pandas()
                    )
                else:  # pandas
                    grouped_data = (
                        df.groupby(plot_feature, observed=True)[
                            [target[idx], weights[idx]]
                        ]
                        .apply(
                            lambda x: helpers._weighted_mean(
                                x.iloc[:, 0], weights=x.iloc[:, 1]
                            )
                        )
                        .reset_index(name=f"{target[idx]}")
                    )

            # mean
            else:  # noqa: PLR5501
                if is_lazy:
                    grouped_data = (
                        df.group_by(plot_feature)
                        .agg(pl.mean(target[idx]).alias(target[idx]))
                        .collect()
                        .to_pandas()
                    )
                else:  # pandas
                    grouped_data = (
                        df.groupby(plot_feature, observed=True)[target[idx]]
                        .mean()
                        .reset_index()
                    )
            grouped_data = grouped_data.sort_values(
                by=plot_feature,
                key=lambda x: x.astype(str)
                if pd.api.types.is_categorical_dtype(x)
                else x,
            )

            # adding trace for current target within the subplot for the feature
            if len(plot_feature) == 2:
                feature_1, feature_2 = plot_feature

                if is_lazy:
                    f1_nunique = df.select(pl.col(feature_1)).collect().n_unique()
                    f2_nunique = df.select(pl.col(feature_2)).collect().n_unique()
                    feature_x = feature_1 if f1_nunique > f2_nunique else feature_2
                else:  # pandas
                    feature_x = (
                        feature_1
                        if df[feature_1].nunique() > df[feature_2].nunique()
                        else feature_2
                    )

                feature_color = feature_2 if feature_x == feature_1 else feature_1
                target_name = f"{target[idx]}_{idx}"
                line_fig = px.line(
                    grouped_data,
                    x=feature_x,
                    y=target[idx],
                    color=feature_color,
                    title=f"{feature_x} vs {feature_color}",
                    labels={feature_x: feature_x, target[idx]: target[idx]},
                )

                for trace in line_fig.data:
                    trace.showlegend = False
                    fig.add_trace(trace, row=row, col=col)
            else:
                feature = plot_feature[0]
                target_name = f"{target[idx]}_{idx}"
                fig.add_trace(
                    go.Scatter(
                        x=grouped_data[feature],
                        y=grouped_data[target[idx]],
                        mode="lines" if not feature == "_aggregate" else "markers",
                        name=target_name,
                        line={"color": color_scale[idx % len(color_scale)]},
                        showlegend=target_name not in legend_added,
                    ),
                    row=row,
                    col=col,
                )
                legend_added.add(target_name)

            # add trace for line
            if add_line:
                line_name = "y=1"
                if pd.api.types.is_numeric_dtype(grouped_data[plot_feature[0]].dtype):
                    add_line_x = [
                        grouped_data[plot_feature[0]].min(),
                        grouped_data[plot_feature[0]].max(),
                    ]
                else:
                    add_line_x = sorted(grouped_data[plot_feature[0]].unique())
                    add_line_x = [add_line_x[0], add_line_x[-1]]
                fig.add_trace(
                    go.Scatter(
                        x=add_line_x,
                        y=[1, 1],
                        mode="lines",
                        line={"dash": "dot", "color": "grey"},
                        name=line_name,
                        showlegend=line_name not in legend_added,
                    ),
                    row=row,
                    col=col,
                )
                legend_added.add(line_name)

    # update layout
    fig.update_layout(
        height=chart_height * num_rows,
        width=chart_width,
        title_text=title,
    )

    return fig


def get_stats(df: Union[pd.DataFrame, pl.LazyFrame], features: list) -> pd.DataFrame:
    """
    Generate summary statistics for the dataset.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.LazyFrame]
        The DataFrame or LazyFrame to use.
    features : list
        The features to use.

    Returns
    -------
    stats : pd.DataFrame
        DataFrame containing summary statistics

    """
    # check if lazy
    is_lazy = isinstance(df, pl.LazyFrame)

    if is_lazy:
        stats = df.select(features).describe().to_pandas()
        stats = stats.set_index("statistic")
        stats = stats.astype(float)
        # get numeric features
        schema = df.collect_schema()
        num_features = [
            feature
            for feature in features
            if schema[feature] in pl.datatypes.group.NUMERIC_DTYPES
        ]
        # add additional statistics (null and zero counts)
        stats.loc["null_pct"] = (
            stats.loc["null_count"] / stats.loc["count"] * 100
        ).round(2)
        exprs = []
        for num_feature in num_features:
            exprs.extend([pl.col(num_feature).eq(0).sum().alias(num_feature)])
        zero_counts = df.select(exprs).collect().to_pandas().iloc[0].to_dict()
        stats.loc["zero_count"] = pd.Series(zero_counts)
        stats.loc["zero_pct"] = (
            stats.loc["zero_count"] / stats.loc["count"] * 100
        ).round(2)

    else:  # pandas
        df_stats = df[features]
        stats = df_stats.describe()
        # add additional statistics
        stats.loc["null_count"] = df_stats.isnull().sum()
        stats.loc["null_pct"] = (df_stats.isnull().sum() / len(df_stats) * 100).round(2)
        stats.loc["zero_pct"] = (df_stats == 0).sum() / len(df_stats) * 100

    # format
    stats = stats.transpose()
    numeric_cols = stats.select_dtypes(include=["float64", "int64"]).columns
    stats[numeric_cols] = stats[numeric_cols].round(2)
    stats = stats.reset_index()
    stats = stats.rename(columns={"index": "Feature"})

    return stats


def get_category_orders(
    df: pd.DataFrame,
    category: str,
    measure: str,
    ascending: bool = False,
    agg: str = "sum",
) -> Dict[str, List[Any]]:
    """
    Get the category order.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to use.
    category : str
        The category to use.
    measure : str
        The measure variable.
    ascending : bool, optional (default=False)
        Whether to sort in ascending order.
    agg : str, optional (default='sum')
        The aggregation function to use.

    Returns
    -------
    category_order : dict
        The category order

    """
    col_order = df.groupby(category)[measure].agg(agg).sort_values(ascending=ascending)
    category_orders = {category: col_order.index.to_list()}

    return category_orders


def _pdp_make_prediction(
    model: Any,
    X: pd.DataFrame,
    x_axis: str,
    x_axis_type: str,
    x_axis_cols: Optional[List[str]],
    value: Any,
    line_color: str,
    line_value: Any,
    quick: bool,
    weights: Optional[pd.Series],
) -> Dict[str, Any]:
    """Make predictions for PDP."""
    X_temp = X.copy()

    # Handle one-hot encoding
    if x_axis_type == "ohe":
        for col in x_axis_cols[1:]:
            X_temp[col] = 0
        # the dropped first column is a reference column and should have a value of 0
        if value != x_axis_cols[0][len(x_axis + "_") :]:
            X_temp[x_axis + "_" + value] = 1
    else:
        X_temp.iloc[:, X_temp.columns.get_loc(x_axis)] = value

    # Set line color values
    if line_color and line_color != "Overall":
        X_temp.iloc[:, X_temp.columns.get_loc(line_color)] = line_value

    # Make predictions
    if quick:
        pred = model.predict(X_temp)[0]
    else:
        pred = model.predict(X_temp)
        pred = helpers._weighted_mean(pred, weights=weights)
    logger.debug(f"predicted color [{line_value}] value [{value}]: {pred}")

    return {
        x_axis: value,
        line_color: line_value,
        "pred": pred,
    }
