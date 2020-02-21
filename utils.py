# -*- coding: utf-8 -*-
#
# Copyright © Spyder Project Contributors
# Licensed under the terms of the MIT License
"""Workshop utility functions."""

# Third-party imports
import matplotlib.pyplot as plt


def plot_correlations(df, size=10):
    """
    Plot a graphical correlation matrix for each pair of columns.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)


def aggregate_by_year(df, date_column, figsize=(15, 8)):
    """
    Return a DataFrame aggregate by year and plot it.

    Input:
        df: pandas DataFrame
        date_column: label of the column with date values
    Return:
        df_yearly: DataFrame grouped by month
    """
    df['year'] = df[date_column].apply(lambda x: x.year)
    df_yearly = df.groupby('year').mean()
    df_yearly .plot(subplots=True, figsize=figsize)

    return df_yearly
