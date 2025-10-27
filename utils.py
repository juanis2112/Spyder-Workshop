# -*- coding: utf-8 -*-
"""Workshop utility functions."""

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np


def plot_correlations(df, size=10):
    """
    Plot a graphical correlation matrix for each pair of columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame on which to calculate correlations.
    size : numeric, optional
        Vertical and horizontal size of the plot. The default is 10.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object of the generated plot.
    ax : matplotlib.axes.Axes
        Axes object of the generated plot.

    """
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    ax.set(xticks=range(len(corr.columns)), xticklabels=corr.columns)
    ax.set(yticks=range(len(corr.columns)), yticklabels=corr.columns)

    plt.show()
    return fig, ax


def aggregate_by_year(df, date_column, figsize=(15, 8)):
    """
    Aggregate a DataFrame by year and plot it.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to aggregate by year and plot.
    date_column : str
        Label of the column containing the date values, as pd.Timestamps.
    figsize : Tuple of (int, int), optional
        Tuple with the size of the figure to generate in (width, height).
        The default is (15, 8).

    Returns
    -------
    df_yearly : pandas.DataFrameGroupBy
        Grouped dataframe by year.

    """
    df['year'] = df[date_column].dt.year
    df_yearly = df.groupby('year').mean(numeric_only=True)
    df_yearly.plot(subplots=True, figsize=figsize)
    return df_yearly


def plot_color_gradients(cmap_category, cmap_list):
    """
    Plot a convention for color gradients used for correlations.

    Parameters
    ----------
    cmap_category : str
        Category of the color map, to use for the plot title.
    cmap_list : list of str
        List of colormap names to plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object of the generated plot.
    ax : matplotlib.axes.Axes
        Axes object of the generated plot.

    """
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, ax = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)

    ax.set_title(cmap_category + ' colormaps', fontsize=14)
    for ax, name in zip([ax], cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))

    xticks = np.linspace(-1, 1, 3)
    xtick_locs = np.linspace(0, 256, 3)

    ax.set(xticks=xtick_locs, xticklabels=xticks)
    ax.get_yaxis().set_visible(False)

    plt.show()
    return fig, ax
