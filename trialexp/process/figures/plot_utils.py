import matplotlib.pylab as plt 

def create_plot_grid(n_plots, ncol, figsize_plot=(3,3), **kwargs):
    nrow = n_plots//ncol + 1
    fig = plt.figure(figsize=(figsize_plot[0]*ncol, figsize_plot[1]*nrow), **kwargs)
    axes = []
    for i in range(n_plots):
        ax = fig.add_subplot(nrow, ncol, i+1)
        axes.append(ax)
    return fig, axes

