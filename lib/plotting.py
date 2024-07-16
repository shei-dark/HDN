import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot(*args, plot_types=None):
    """
    This function accepts a variable number of inputs in the form of tuples and plots them in subplots.
    Each input tuple is expected to be in the form (data, title, extra).
    - data: list or array-like object to plot or list of such objects for multiple scatter plots
    - title: string, title of the subplot
    - extra: tuple for specific customization (e.g., centre square) or None
    - plot_types: list of strings, specifying the plot type for each subplot ('imshow' or 'scatter')
    """

    n = len(args)
    cols = 2
    rows = (n + 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axs = axs.flatten()

    for i, ((data, title, extra), plot_type) in enumerate(zip(args, plot_types)):
        if plot_type == 'imshow':
            axs[i].imshow(data)
        elif plot_type == 'scatter':
            if isinstance(data, list):
                for scatter_data, label in zip(data, extra):
                    axs[i].scatter(scatter_data[:, 0], scatter_data[:, 1], label=label[0], s=label[1])
            else:
                axs[i].scatter(data[:, 0], data[:, 1], label=extra)
            axs[i].legend(loc='best')
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

    plt.tight_layout()
    plt.show()

# def scat(ax, x, y, color, size, lbl, alpha=1, edgecolors="none"):
#     ax.scatter(
#     x,
#     y,
#     c=color,
#     s=size,
#     label=lbl,
#     alpha=alpha,
#     edgecolors=edgecolors,
#     )

# def plot(*args):

#     """
#     This function accepts a variable number of inputs in the form of tuples and plots them in subplots.
#     Each input tuple is expected to be in the form (data, title, boolean).
#     - data: list or array-like object to plot
#     - title: string, title of the subplot
#     - boolean: boolean, to apply specific customization (e.g., centre square)
#     """

#     n = len(args)
#     cols = 2
#     rows = (n+1) // cols

#     fig, axs = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
#     axs = axs.flatten()

#     for i, (data, title, _) in enumerate(args):
#         axs[i].imshow(data)
#         axs[i].set_title(title)
#         axs[i].set_xticks([])  # Hide x-axis ticks
#         axs[i].set_yticks([])  # Hide y-axis ticks
#         if _:
#             y, x , size = _[0], _[1], _[2]
#             rect = patches.Rectangle(
#                 (x, y),
#                 size,
#                 size,
#                 linewidth=1,
#                 edgecolor="r",
#                 facecolor="none",
#             )
#             axs[i].add_patch(rect)
    
#     for j in range(i+1, len(axs)):
#         fig.delaxes(axs[j])

#     plt.tight_layout()
#     plt.show()