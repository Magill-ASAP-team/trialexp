import matplotlib.pylab as plt 

def create_plot_grid(n_plots, ncol, figsize_plot=(3,3)):
    ncol = 4
    nrow = n_plots//4 + 1
    fig = plt.figure(figsize=(figsize_plot[0]*ncol, figsize_plot[1]*nrow))
    axes = []
    for i in range(n_plots):
        ax = fig.add_subplot(nrow, ncol, i+1)
        axes.append(ax)
    return fig, axes