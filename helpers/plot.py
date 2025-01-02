import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def set_style(style='whitegrid', context='paper', font_scale=1.2):
    """
    Set consistent style for plots
    
    Args:
        style (str): Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark')
        context (str): Context ('paper', 'talk', 'poster')
        font_scale (float): Scale factor for font sizes
    """
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)

def plot_distribution(data, title=None, xlabel=None, ylabel='Count'):
    """
    Create a distribution plot with both histogram and KDE
    
    Args:
        data (array-like): Data to plot
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, kde=True)
    if title:
        plt.title(title)
    plt.xlabel(xlabel if xlabel else '')
    plt.ylabel(ylabel)
    return plt.gcf()
