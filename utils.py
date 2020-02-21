# -*- coding: utf-8 -*-
#
# Copyright © Spyder Project Contributors
# Licensed under the terms of the MIT License
"""Workshop utility functions."""

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np


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


def predicted_temperature(humidity, beta_0, beta_1):
    """
    Return a predicted temperature based on the humidity.

    Uses a linear regression as model: temperature = beta_0 + beta_1 * humidity
    """
    return beta_0 + beta_1 * humidity

cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma',])]



def plot_color_gradients(cmap_category, cmap_list):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
    fig, axes = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)

    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)
    
    print(axes)
    print(cmap_list)
    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        ax.text(-.01, .5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()


for cmap_category, cmap_list in cmaps:
    plot_color_gradients(cmap_category, cmap_list)

plt.show()
