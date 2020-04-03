import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_distribution(data, title, x_axis_name, x_axis_data, y_axis_name, y_axis_data, x_lim=1e4, y_lim=1e5):
    """
    Plots given data.
    :param data: data to be plotted
    :type data: pandas.DataFrame
    :param title: title of the plot
    :type title: str
    :param x_axis_name: name of the x axis
    :type x_axis_name: str
    :param x_axis_data: name of the data to be plotted along x axis
    :type x_axis_data: str
    :param y_axis_name: name of the y axis
    :type y_axis_name: str
    :param y_axis_data: name of the data to be plotted along y axis
    :type y_axis_data: str
    """
    sns.set(style='darkgrid', context='poster', font='Verdana')
    f, ax = plt.subplots()
    ax.set(xscale='log', yscale='log')
    plt.xlim(1e-1, x_lim)
    plt.ylim(1e-1, y_lim)
    ax.grid(False)
    sns.regplot(x_axis_data, y_axis_data, data, ax=ax, scatter_kws={'s': 500}, fit_reg=False, color='xkcd:sage')
    ax.set_ylabel(y_axis_name)
    ax.set_xlabel(x_axis_name)
    ax.grid(False)
    plt.title(title)
    plt.show()


def plot_length_distribution(lengths_file, x_lim, y_lim):
    """
    Plots the distribution of sentences length.
    :param lengths_file: name of the file containing sentences lengths
    :type lengths_file: str
    """
    lengths = pd.read_csv(lengths_file, sep=',', index_col=0).get_values().flatten()
    distribution_array = lengths
    distribution = np.zeros(np.max(distribution_array) + 1, dtype=np.float32)
    distribution_index = np.array(range(np.max(distribution_array) + 1), dtype=np.float32)
    for d in distribution_array:
        distribution[d] += 1
    data = pd.DataFrame()
    data['Sentence length'] = distribution_index
    data['Number of sentences'] = distribution
    plot_distribution(data, 'Распределба на должина на реченици', 'Должина на реченица', 'Sentence length',
                      'Број на реченици', 'Number of sentences', x_lim, y_lim)
