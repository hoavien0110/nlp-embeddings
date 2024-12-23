import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np



def plot_discrete_pdf(pdf, params, x, ax, **kwargs):
    """
    Plot the probability density function (PDF) of a discrete distribution.
    Parameters:
    pdf: function
        The PDF function to be plotted.
    params: dict
        A dictionary of parameters to be passed to the PDF function.
    x: list
        A list of values for which the PDF will be calculated.
    ax: matplotlib.axes.Axes
        The axes on which the PDF will be plotted.
    Returns:
    None
    """
    # bar 
    sns.barplot(x=x, y=pdf(x, **params), ax=ax, **kwargs)
    plt.show()