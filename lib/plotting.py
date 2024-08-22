import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np


def plot_wo_b(*args, plot_types=None, box_size=3):
    """
    This function accepts a variable number of inputs in the form of tuples and plots them in subplots.
    Each input tuple is expected to be in the form (data, title, extra).
    - data: list or array-like object to plot or list of such objects for multiple scatter plots
    - title: string, title of the subplot
    - extra: tuple for specific customization (e.g., centre square) or None
    - plot_types: list of strings, specifying the plot type for each subplot ('imshow' or 'scatter')
    """
    colors = [
    (0, 0, 0),           # Black
    (230/255, 159/255, 0),  # Orange
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353), # violet
    (0, 158/255, 115/255),  # Bluish green
    ]

    # Create the colormap
    cmap = mcolors.ListedColormap(colors)

    # Define the bounds and normalization
    bounds = np.arange(len(colors)+1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    n = len(args)
    cols = 2
    rows = (n + 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(box_size * cols, box_size * rows))
    axs = axs.flatten()

    for i, ((data, title, extra), plot_type) in enumerate(zip(args, plot_types)):
        if plot_type == 'imshow_l':
            axs[i].imshow(data, cmap=cmap, norm=norm)
        elif plot_type == 'imshow':
            axs[i].imshow(data)
        elif plot_type == 'scatter':
            for scatter_data, label in zip(data, extra):
                axs[i].scatter(scatter_data[:, 0], scatter_data[:, 1], c=label[0], cmap=cmap, norm=norm, s=label[1])
            last_data = data[-1]
            last_label = extra[-1]
            axs[i].scatter(last_data[:, 0], last_data[:, 1], facecolors='none', edgecolors='red', s=last_label[1]*1.5, linewidth=1)

            # Custom legend for the scatter plot
            
            handle = plt.Line2D([0], [0], marker='o', color='r', markerfacecolor=cmap(norm(last_label[0])),\
                                markersize=7, linestyle='None', markeredgewidth=1, label='Test')
            axs[i].legend(handles=[handle], loc='best')

        axs[i].set_title(title)
        axs[i].set_xticks([])  # Hide x-axis ticks
        axs[i].set_yticks([])  # Hide y-axis ticks
        if isinstance(extra, tuple):
            y, x, size = extra
            rect = patches.Rectangle(
                (x, y),
                size,
                size,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            axs[i].add_patch(rect)

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    # Add custom legend to the top center of the whole figure
    labels = [
        "Background", "Golgi", "Mitochondria", "Granules"
    ]
    handles = [patches.Patch(color=color, label=label) for color, label in zip(colors, labels)]

    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))


    plt.tight_layout()
    plt.show()

def plot_w_b(*args, plot_types=None, box_size=3, with_test=False):
    """
    This function accepts a variable number of inputs in the form of tuples and plots them in subplots.
    Each input tuple is expected to be in the form (data, title, extra).
    - data: list or array-like object to plot or list of such objects for multiple scatter plots
    - title: string, title of the subplot
    - extra: tuple for specific customization (e.g., centre square) or None
    - plot_types: list of strings, specifying the plot type for each subplot ('imshow' or 'scatter')
    """
    colors = [
        (0.8, 0.8, 0.8),               # white for label -1
        (1, 0.8, 0.9),          # pink for label 0
        (230/255, 159/255, 0),  # orange for label 1
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353), # violet for label 2
        (0, 158/255, 115/255),  # bluish green for label 3
        
    ]

    # Create the colormap
    cmap = mcolors.ListedColormap(colors)
    # Define the bounds and normalization
    bounds = [-1, 0, 1, 2, 3, 4]    
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    n = len(args)
    cols = 2
    rows = (n + 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(box_size * cols, box_size * rows))
    axs = axs.flatten()

    for i, ((data, title, extra), plot_type) in enumerate(zip(args, plot_types)):
        if plot_type == 'imshow_l':
            axs[i].imshow(data, cmap=cmap, norm=norm)
        elif plot_type == 'imshow':
            axs[i].imshow(data)
        elif plot_type == 'scatter':
            for scatter_data, label in zip(data, extra):
                axs[i].scatter(scatter_data[:, 0], scatter_data[:, 1], c=label[0], cmap=cmap, norm=norm, s=label[1], alpha=0.3)
            if with_test:
                last_data = data[-1]
                last_label = extra[-1]
                axs[i].scatter(last_data[:, 0], last_data[:, 1], facecolors='none', edgecolors='red', s=last_label[1]*1.5, linewidth=1)

                # Custom legend for the scatter plot
                
                handle = plt.Line2D([0], [0], marker='o', color='r', markerfacecolor=cmap(norm(last_label[0])),\
                                    markersize=7, linestyle='None', markeredgewidth=1, label='Test')
                axs[i].legend(handles=[handle], loc='best')

        axs[i].set_title(title)
        axs[i].set_xticks([])  # Hide x-axis ticks
        axs[i].set_yticks([])  # Hide y-axis ticks
        if isinstance(extra, tuple):
            try:
                y, x, height, width = extra
            except ValueError:
                y, x, height = extra
                width = height
            rect = patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            axs[i].add_patch(rect)

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    # Add custom legend to the top center of the whole figure
    labels = [
        "Outside of the cell", "Uncategorized", "Nucleus", "Granules", "Mitochondria"
    ]
    handles = [patches.Patch(color=color, label=label) for color, label in zip(colors, labels)]

    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))


    plt.tight_layout()
    plt.show()
    return plt
