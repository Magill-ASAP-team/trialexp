from scipy.cluster.hierarchy import dendrogram
import matplotlib.pylab as plt 
import numpy as np

def create_plot_grid(n_plots, ncol, figsize_plot=(3,3), **kwargs):
    nrow = n_plots//ncol + 1
    fig = plt.figure(figsize=(figsize_plot[0]*ncol, figsize_plot[1]*nrow), **kwargs)
    axes = []
    for i in range(n_plots):
        ax = fig.add_subplot(nrow, ncol, i+1)
        axes.append(ax)
    return fig, axes

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)