import os
from matplotlib import colors
import glob
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import natsort
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import pingouin as pg
import bct
from neuromaps import images
import subprocess
from datetime import datetime
import pingouin as pg
import nibabel as nib
from brainspace.mesh.mesh_io import read_surface
from surfplot import Plot
import matplotlib.pyplot as plt
import seaborn as sns
import cmasher as cmr
from nilearn.plotting import cm
from brainspace.utils.parcellation import map_to_labels
from surfplot import Plot
from surfplot.utils import add_fslr_medial_wall
pjoin = os.path.join

class Config(object):
    resources = "Users/keannarowchan/Desktop/statistical-manifolds/resources"
    atlas = "/Users/keannarowchan/Desktop/statistical-manifolds/atlases/Schaefer2018_1000Parcels_7Networks_order.dlabel.nii"
    gradients = "/Users/keannarowchan/Desktop/statistical-manifolds/data/pca-gradients-centered/"
    figures = "/Users/keannarowchan/Desktop/statistical-manifolds/figures"
    k = 3

FIG_DIR = "/Users/keannarowchan/Desktop/statistical-manifolds/figures/reference/"

def load_table(fname):
    return pd.read_table(fname, index_col=0)


def load_gradients(fname, k=None):
    df = pd.read_table(fname, index_col=0)
    if k is not None:
        df = df.iloc[:, :k]
        
    df = df.reset_index().rename(columns={'index': 'roi'})
    df = parse_roi_names(df)
    return df

def get_surfaces(style='inflated', load=True):
    config = Config()
    surf_path = os.path.join(config.resources, 'surfaces')
    surfaces = get_files([surf_path, f'*.{style}_*'])
    
    if load:
        surfs = [read_surface(i) for i in surfaces]
        return dict(zip(['lh', 'rh'], surfs))
    else:
        return surfaces

def set_plotting():
    plt.rcParams["savefig.format"] = 'png'
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = 'tight'
    sns.set_context('paper')

def plot_cbar(cmap, vmin, vmax, orientation='vertical', size=None, n_ticks=2, decimals=2, fontsize=12, show_outline=False, as_int=False):

    if size is None and (orientation == 'vertical'):
        size = (.3, 4)
    if size is None and (orientation == 'horizontal'):
        size = (4, .3)

    x = np.array([[0,1]])
    plt.figure(figsize=size)
    img = plt.imshow(x, cmap=cmap)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])

    ticks = np.linspace(0, 1, n_ticks)
    tick_labels = np.around(np.linspace(vmin, vmax, n_ticks), decimals)
    cbar = plt.colorbar(orientation=orientation, cax=cax, ticks=ticks)
    if as_int:
        cbar.set_ticklabels(tick_labels.astype(int))
    else:
        cbar.set_ticklabels(tick_labels)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    if not show_outline:
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=0)
    return cbar

def weights_to_vertices(data, target, labels=None):
    if isinstance(target, str): 
        vertices = nib.load(target).get_fdata().ravel()
    else:
        vertices = target.ravel()

    if labels is not None:
        data = _align_labels_to_atlas(data, labels, vertices)

    mask = vertices != 0
    map_args = dict(target_lab=vertices, mask=mask, fill=np.nan)
    if (len(data.shape) == 1) or (data.shape[1] == 1):
        weights = map_to_labels(data.ravel(),  **map_args)
    else:
        weights = [map_to_labels(x, **map_args) for x in data.T]
    return weights

def plot_3d(x, y, z, color=None, ax=None, view_3d=(35, -110), **kwargs):
    if ax is None:
        fig = plt.figure(figsize=(4, 4), frameon=False)
        ax = fig.add_subplot(projection='3d')
    
    ax.scatter(xs=x, ys=y, zs=z, c=color, **kwargs)
    ax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3')
    if view_3d is not None:
        ax.view_init(elev=view_3d[0], azim=view_3d[1])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    return ax

def yeo_cmap(as_palette=False, networks=7):
    if networks == 17:
        cmap = {
            'VisCent': (120, 18, 136),
            'VisPeri': (255, 0, 2),
            'SomMotA': (70, 130, 181),
            'SomMotB': (43, 204, 165),
            'DorsAttnA': (74, 156, 61),
            'DorsAttnB': (0, 118, 17),
            'SalVentAttnA': (196, 58, 251),
            'SalVentAttnB': (255, 153, 214),
            'TempPar': (9, 41, 250),
            'ContA': (230, 148, 36),
            'ContB': (136, 50, 75),
            'ContC': (119, 140, 179),
            'DefaultA': (255, 254, 1),
            'DefaultB': (205, 62, 81),
            'DefaultC': (0, 0, 132),
            'LimbicA': (224, 248, 166),
            'LimbicB': (126, 135, 55)
        }
    else:
        cmap = {
            'Vis': (119, 20, 140),
            'SomMot': (70, 126, 175), 
            'DorsAttn': (0, 117, 7), 
            'SalVentAttn': (195, 59, 255), 
            'Limbic': (219, 249, 165), 
            'Cont': (230, 149, 33), 
            'Default': (205, 65, 80) 
        }
    cmap = {k: np.array(v) / 255 for k, v in cmap.items()}
    if as_palette:
        return sns.color_palette(cmap.values())
    else:
        return cmap

def plot_eigenvalues():
    fname = "/Users/keannarowchan/Desktop/statistical-manifolds/data/reference_eigenvalues.tsv"
    eigenvals = load_table(fname)['proportion'][:20]
    k = len(eigenvals)

    fig, ax = plt.subplots( figsize=(2, 2.7))
    ax.plot(np.arange(k) + 1, eigenvals, marker='o', markerfacecolor='w', color='k')
    ax.set(ylabel='% Variance expl.', xlabel='Principal\ncomponent (PC)', xticks=np.arange(0, 21, 5), yticks=np.arange(0, .26, .05))
    
    sns.despine()
    fig.tight_layout(w_pad=3)
    return fig

def show_eigenvalues():
    config = Config()
    fname = "/Users/keannarowchan/Desktop/statistical-manifolds/data/reference_eigenvalues.tsv"
    df = load_table(fname)
    df['cumulative'] = df['proportion'].cumsum()
    return df

def plot_ref_brain_gradients(k=3):
    config = Config()
    fname = "/Users/keannarowchan/Desktop/statistical-manifolds/data/reference_gradient.tsv"
    gradients = load_gradients(fname, k)
    prefix = os.path.join(FIG_DIR, 'gradients_')
    cmap = "twilight_shifted"

    grads = gradients.filter(like='g').values
    vmax = np.around(grads.max(), decimals=1)
    plotting.plot_cbar(cmap, -vmax, vmax, 'horizontal', size=(1, .3), n_ticks=3)
    plt.savefig(prefix + 'cbar')        
    
    surfaces = get_surfaces()
    for i in range(k):
        x = plotting.weights_to_vertices(grads[:, i], Config().atlas)

        p = Plot(surfaces['lh'], surfaces['rh'])
        p.add_layer(x, cmap=cmap, color_range=(-vmax, vmax), cbar=False)
        fig = p.build(colorbar=False)
        fig.savefig(prefix + f'PC{i + 1}_brain')

def plot_loading_distributions(k=3, view_3d=(30, -110)):
    config = Config()
    fname = "/Users/keannarowchan/Desktop/statistical-manifolds/data/reference_gradient.tsv"
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
        ax1 = fig.add_subplot(gs[:, :6], projection='3d')
        plotting.plot_3d(df['g1'], df['g2'], df['g3'], color=df['c'], ax=ax1, view_3d=view_3d, s=10, lw=0)

        ax2 = fig.add_subplot(gs[:5, 6:9])
        sns.scatterplot(x='g1', y='g2', hue='network', data=df, palette=cmap, legend=False, ax=ax2, linewidths=0, s=8, edgecolor='none')
        ax2.set(xlabel='', xticklabels=[], ylabel='PC2', xlim=(-2.5, 4), ylim=(-3, 4))
        ax2 = fig.add_subplot(gs[7:, 6:9])
        sns.scatterplot(x='g1', y='g3', hue='network', data=df, palette=cmap, legend=False, ax=ax2, linewidths=0, s=8, edgecolor='none')
        ax2.set(xlabel='PC1', ylabel='PC3', xlim=(-2.5, 4), ylim=(-2.5, 3))

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

    if k == 4:
        fig, ax = plt.subplots(figsize=(3, 1.5))
        sns.kdeplot(x='g4', hue='network', data=df, palette=cmap, shade=True, ax=ax, legend=False, alpha=.7)
        
        ymax = .12
        ax.set(xlabel='Loading', ylim=(0, ymax), yticks=(0, ymax), ylabel='', yticklabels=(['0', ymax]))
        sns.despine()
        return fig

def plot_eccentricity_calc(view_3d=(30, -110)):
    config = Config()
    fname = "/Users/keannarowchan/Desktop/statistical-manifolds/data/reference_gradient.tsv"
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
    fname = "/Users/keannarowchan/Desktop/statistical-manifolds/data/reference_gradient.tsv"
    df = load_gradients(fname, k)
    grads = df.filter(like='g').values
    
    centroid = np.mean(grads, axis=0)

    ecc = np.linalg.norm(grads - centroid, axis=1)
    vmax = np.nanmax(ecc)
    vmin = np.nanmin(ecc)
    cmap = 'viridis'

    prefix = os.path.join(FIG_DIR, 'ref_ecc_')
    if k == 2:
        fig, ax = plt.subplots()
        ax = sns.scatterplot('g1', 'g2', c=ecc, data=df, palette=cmap, 
                             ax=ax)
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

    surfaces = get_surfaces()
    x = plotting.weights_to_vertices(ecc, Config().atlas)

    p = Plot(surfaces['lh'], surfaces['rh'])
    p.add_layer(x, color_range=(vmin, vmax), cmap=cmap)
    cbar_kws = dict(location='bottom', decimals=2, fontsize=12, n_ticks=2, shrink=.4, aspect=4, draw_border=False, pad=-.06)
    fig = p.build(cbar_kws=cbar_kws)
    fig.savefig(prefix + 'brain')

    return pd.DataFrame({'roi': df['roi'], 'distance': ecc})

def main():
    
    config = Config()
    FIG_DIR = "/Users/keannarowchan/Desktop/statistical-manifolds/figures/reference/"
    df = show_eigenvalues()

    fig = plot_eigenvalues()
    fig.savefig(os.path.join("/Users/keannarowchan/Desktop/statistical-manifolds/figures/reference/", 'var_explained'))
    
    plot_ref_brain_gradients(config.k)
    plot_eccentricity_calc()
    
    ecc = reference_eccentricity(config.k)
    
    fig = plot_loading_distributions(k=config.k)
    fig.savefig(os.path.join("/Users/keannarowchan/Desktop/statistical-manifolds/figures/reference/", 'ref_networks'))


