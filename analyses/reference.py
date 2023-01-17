import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmasher as cmr
from surfplot import Plot
import warnings
warnings.filterwarnings('ignore')

from utils import load_table, load_gradients, get_surfaces
from config import Config
import plotting

FIG_DIR = os.path.join(Config().figures, 'reference')
os.makedirs(FIG_DIR, exist_ok=True)

plotting.set_plotting()

def plot_eigenvalues():
    config = Config()
    fname = os.path.join(config.gradients, 'reference_eigenvalues.tsv')
    eigenvals = load_table(fname)['proportion'][:20]
    k = len(eigenvals)

    fig, ax = plt.subplots( figsize=(2, 2.7))
    ax.plot(np.arange(k) + 1, eigenvals, marker='o', markerfacecolor='w', color='k')
    ax.set(ylabel='% Variance expl.', xlabel='Principal\ncomponent (PC)', xticks=np.arange(0, 21, 5), yticks=np.arange(0, .26, .05))
    
    sns.despine()
    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(FIG_DIR, 'plotted_eigenvalues'))
    return fig


def show_eigenvalues():
    config = Config()
    fname = os.path.join(config.gradients, 'reference_eigenvalues.tsv')
    df = load_table(fname)
    df['cumulative'] = df['proportion'].cumsum()
    return df


def plot_ref_brain_gradients(k=3):
    config = Config()
    fname = os.path.join(config.gradients, 'reference_gradient.tsv')
    gradients = load_gradients(fname, k)
    prefix = os.path.join(FIG_DIR, 'gradients_')
    cmap = 'twilight_shifted'

    grads = gradients.filter(like='g').values
    vmax = np.around(grads.max(), decimals=1)
    plotting.plot_cbar(cmap, -vmax, vmax, 'horizontal', size=(1, .3), n_ticks=3)
    plt.savefig(prefix + 'cbar')        
    
    surfaces = get_surfaces()
    for i in range(k):
        x = plotting.weights_to_vertices(grads[:, i], config.atlas)

        p = Plot(surfaces['lh'], surfaces['rh'])
        p.add_layer(x, cmap=cmap, color_range=(-vmax, vmax), cbar=False)
        fig = p.build(colorbar=False)
        fig.savefig(prefix + f'PC{i + 1}_brain')

def plot_loading_distributions(k=3, view_3d=(30, -110)):
    config = Config()
    fname = os.path.join(config.gradients, 'reference_gradient.tsv')
    df = load_gradients(fname, k)

    cmap = plotting.yeo_cmap()
    if k == 2:
        g = sns.jointplot(x='g1', y='g2', hue='network', data=df, palette=cmap, legend=False, height=4.5, marginal_kws=dict(alpha=.7), joint_kws=dict(linewidth=0, s=15),)
        g.ax_joint.set(xlabel='PC1', ylabel='PC2')
        return g

    if k == 3:
        df['c'] = df['network'].apply(lambda x: cmap[x])

        fig = plt.figure(figsize=(10, 4))
        gs = fig.add_gridspec(nrows=12, ncols=12) 
        # 3D scatter plot
        ax1 = fig.add_subplot(gs[:, :6], projection='3d')
        plotting.plot_3d(df['g1'], df['g2'], df['g3'], color=df['c'], ax=ax1, view_3d=view_3d, s=10, lw=0)

        # 2D views
        ax2 = fig.add_subplot(gs[:5, 6:9])
        sns.scatterplot(x='g1', y='g2', hue='network', data=df, palette=cmap, legend=False, ax=ax2, linewidths=0, s=8, edgecolor='none')
        ax2.set(xlabel='', xticklabels=[], ylabel='PC2', xlim=(-6, 10), ylim=(8, -8))
        ax2 = fig.add_subplot(gs[7:, 6:9])
        sns.scatterplot(x='g1', y='g3', hue='network', data=df, palette=cmap, legend=False, ax=ax2, linewidths=0, s=8, edgecolor='none')
        ax2.set(xlabel='PC1', ylabel='PC3', xlim=(-6, 10), ylim=(8, -8))

        # distribution plots
        for i, g in zip([1, 5, 9], ['g1', 'g2', 'g3']):
            ax = fig.add_subplot(gs[i:i+2, 10:])
            sns.kdeplot(x=g, hue='network', data=df, palette=cmap, shade=True, ax=ax, legend=False, alpha=.7)
            
            ax.set(xlabel='', ylim=(0, .25), yticks=(0, .25), ylabel='', xlim=(-4, 4), xticks=range(-4, 5, 2), yticklabels=(['0', .25]))
            if g == 'g3':
                ax.set_xlabel('Loading')
            else:
                ax.set_xticklabels([])
            num = g[1]
            ax.set_title(f'PC{num}', loc='right', y=.5)
            sns.despine()
        
        return fig
        #fig.savefig(os.path.join(FIG_DIR, 'loading_distributions'))


def plot_eccentricity_calc(view_3d=(30, -110)):
    config = Config()
    fname = os.path.join(config.gradients, 'reference_gradient.tsv')
    df = load_gradients(fname, 3)

    cmap = plotting.yeo_cmap()
    df['c'] = df['network'].apply(lambda x: cmap[x])

    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(projection='3d')
    plotting.plot_3d(df['g1'], df['g2'], df['g3'], color=df['c'], ax=ax1, view_3d=view_3d, s=10, lw=0, alpha=.2)

    centroid = np.array([0, 0, 0])
    for i in [10, 212, 100, 440]:
        data = np.vstack([centroid, df.loc[i, ['g1', 'g2', 'g3']].values])
        ax1.plot(data[:, 0], data[:, 1], data[:, 2], c='k', ls='--')
        ax1.scatter([0],[0], [0], color='k', marker='s', s=30, alpha=1)
        plotting.plot_3d(data[1, 0], data[1, 1], data[1, 2], df.loc[i, 'c'], ax=ax1, alpha=1, edgecolor='k', view_3d=view_3d)
        plotting.plot_3d(data[0, 0], data[0, 1], data[0, 2], color='k', ax=ax1, alpha=1, view_3d=view_3d)
    fig.savefig(os.path.join(FIG_DIR, 'ecc_calculation'))


def reference_eccentricity(k=3, view_3d=(30, -110)):
    config = Config()
    fname = os.path.join(config.gradients, 'reference_gradient.tsv')
    df = load_gradients(fname, k)
    grads = df.filter(like='g').values
    
    centroid = np.mean(grads, axis=0)
    # should be 0, 0, 0
    assert np.allclose(np.around(centroid, decimals=16), np.zeros(k))

    ecc = np.linalg.norm(grads - centroid, axis=1)
    vmax = np.nanmax(ecc)
    vmin = np.nanmin(ecc)
    cmap = 'viridis'

    prefix = os.path.join(FIG_DIR, 'ref_ecc_')
    if k == 2:
        fig, ax = plt.subplots()
        ax = sns.scatterplot('g1', 'g2', c=ecc, data=df, palette=cmap, ax=ax)
        ax.set(xlabel='PC1', ylabel='PC2', xlim=(-3, 4), ylim=(-3, 4))
        sns.despine()
        fig.savefig(prefix + 'scatter')
    elif k == 3:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        plotting.plot_3d(grads[:, 0], grads[:, 1], grads[:, 2], color=ecc, ax=ax, cmap=cmap, lw=0, s=10, view_3d=view_3d)
        fig.savefig(prefix + 'scatter')
    
    plotting.plot_cbar(cmap, vmin, vmax, size=(.2, 1), fontsize=8)
    plt.savefig(prefix + 'cbar')      

    #surfaces = get_surfaces()
    #x = plotting.weights_to_vertices(ecc, Config().atlas)

    #p = Plot(surfaces['lh'], surfaces['rh'])
    #p.add_layer(x, color_range=(vmin, vmax), cmap=cmap)
    #cbar_kws = dict(location='bottom', decimals=2, fontsize=12, n_ticks=2, shrink=.4, aspect=4, draw_border=False, pad=-.06)
    #fig = p.build(cbar_kws=cbar_kws)
    #fig.savefig(prefix + 'brain')

    return pd.DataFrame({'roi': df['roi'], 'distance': ecc})


def main():
    
    config = Config()

    df = show_eigenvalues()
    df.to_csv(os.path.join(config.results, 'ref_eigenvalues.tsv'), sep='\t', index=False)

    fig = plot_eigenvalues()
    fig.savefig(os.path.join(FIG_DIR, 'var_explained'))
    plot_eigenvalues()

    #plot_ref_brain_gradients(config.k)
    plot_loading_distributions()
    plot_eccentricity_calc()
    
    ecc = reference_eccentricity(config.k)
    ecc.to_csv(os.path.join(config.results, f'ref_ecc.tsv'), sep='\t', index=False)
    
    fig = plot_loading_distributions(k=config.k)
    fig.savefig(os.path.join(FIG_DIR, 'ref_networks'))

main()
#if __name__ == '__main__':
#    main()
