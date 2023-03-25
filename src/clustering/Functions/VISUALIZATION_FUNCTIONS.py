import matplotlib.pyplot as plt
from matplotlib import ticker
import mpl_toolkits.mplot3d 
from sklearn import manifold
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import pandas as pd
import numpy as np


def t_sne(n_components, perplexity, IVs, DVs, show_plot):
    
    sr_tsne = manifold.TSNE(
      n_components=n_components, learning_rate="auto", 
      perplexity=perplexity, init="pca", random_state=123
    ).fit_transform(IVs)

    if show_plot == True:
        fig, axs = plt.subplots(figsize=(15, 12), nrows=1)

        im = axs.scatter(sr_tsne[:, 0], sr_tsne[:, 1], c=DVs)
        _ = axs.set_title("t-SNE Embedding of Costa Rica Bands Data", fontsize = 28)

        plt.xlabel('Embedding Dimension 1', fontsize = 23)
        plt.ylabel('Embedding Dimension 2', fontsize = 23);

        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax);
        
    else:
        print('No Print')

    return sr_tsne



def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(10, 10),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()
