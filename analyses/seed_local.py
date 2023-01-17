import os
from matplotlib import colors
import glob
import warnings
warnings.filterwarnings('ignore')
import natsort
from matplotlib.colors import LinearSegmentedColormap
import cmasher as cmr
import bct
from neuromaps import images
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import pingouin as pg
import nibabel as nib
from brainspace.mesh.mesh_io import read_surface
from surfplot import Plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.plotting import cm
from brainspace.utils.parcellation import map_to_labels
from surfplot.utils import add_fslr_medial_wall
pjoin = os.path.join

class Config(object):

    atlas = "/Users/keannarowchan/Desktop/server_to_desktop/Schaefer2018_1000Parcels_7Networks_order.dlabel.nii"
    k = 3

def _align_labels_to_atlas(x, source_labels, target_labels):

    target = np.unique(target_labels)[1:]
    df1 = pd.DataFrame(target, index=target)
    df2 = pd.DataFrame(x, index=source_labels)
    return pd.concat([df1, df2], axis=1).iloc[:, 1:].values

def get_surfaces(style='inflated', load=True):
    config = Config()
    surf_path = "/Users/keannarowchan/Desktop/server_to_desktop/surfaces/"
    surfaces = get_files([surf_path, f'*.{style}_*'])
    
    if load:
        surfs = [read_surface(i) for i in surfaces]
        return dict(zip(['lh', 'rh'], surfs))
    else:
        return surfaces

def set_plotting():
    #plt.rcParams['font.family'] = ['Arial']
    plt.rcParams["savefig.format"] = 'png'
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = 'tight'
    sns.set_context('paper')

def get_sulc():
    config = Config()
    surf_path = "/Users/keannarowchan/Desktop/server_to_desktop/surfaces"
    img = os.path.join(surf_path, 'S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii')
    vertices = nib.load(img).get_fdata().ravel()
    return add_fslr_medial_wall(vertices)

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

set_plotting()

def parse_roi_names(x, col='roi'):
    roi_cols = ['hemi', 'network', 'name']
    x[roi_cols] = x[col].str.split('_', n=3, expand=True).iloc[:, 1:]
    return x

def fdr_correct(x, colname='p-unc'):
    """Apply FDR correction across all rows of dataframe"""
    corrected = pg.multicomp(x[colname].values, method='fdr_bh')
    x[['sig_corrected', 'p_fdr']] = np.array(corrected).T
    return x

def schaefer1000_roi_ix():
    x = np.arange(1000) + 1
    return x[~np.isin(x, [533, 903])]

def get_files(pattern, force_list=False):
    if isinstance(pattern, list):
        pattern = pjoin(*pattern)
    
    files = natsort.natsorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError('Pattern could not detect file(s)')

    if (not force_list) & (len(files) == 1):
        return files[0] 
    else:
        return files

def find_cluster_seed(df, method='middle'):
    if method == 'max':
        return df.loc[df['T'].idxmax()]
    elif method == 'middle':
        return df.loc[np.abs(df['T'] - df['T'].mean()).idxmin()]


def connect_seed(cmats, seed_region):
    list_ = []
    for i in cmats:
        cmat = pd.read_table(i, index_col=0)
        res = pd.DataFrame(cmat.loc[seed_region].reset_index().values, columns=['roi', 'r'])
        
        res['roi_ix'] = schaefer1000_roi_ix()

        fname = os.path.basename(i)
        res['sub'] = fname.split('_')[0]
        res['epoch'] = fname.split('_')[-1].split('.')[0]
        list_.append(res)

    connectivity = pd.concat(list_)
    connectivity['r'] = connectivity['r'].astype(np.float)
    return connectivity


def seed_analysis(contrasts, clust_num, cmats, epochs):
    seed = find_cluster_seed(contrasts.query("cluster == @clust_num"))
    connectivity = connect_seed(cmats, seed['roi'])

    df = connectivity.query("epoch in @epochs")
    res = df.groupby(['roi', 'roi_ix'], sort=False) \
            .apply(pg.pairwise_ttests, dv='r', within='epoch', subject='sub') \
            .reset_index()
    res['T'] = -res['T']
    res['sig'] = (res['p-unc'] < .05).astype(float)

    # get network-level changes
    df = parse_roi_names(df)
    networks = df.groupby(['sub', 'network', 'epoch']) \
                 .mean() \
                 .reset_index()
    return fdr_correct(res), seed, networks


def plot_seed_map(data, seed_region, sig_style=None, use_fdr=True, seed_color="Greens", show_left_vis=False):
    if use_fdr:
        sig_regions = data.query("sig_corrected == 1")
    else:
        sig_regions = data.query("sig == 1")
    
    seed = (data['roi'] == seed_region).astype(float)
    
    x = weights_to_vertices(data['T'].astype(float).values, Config().atlas, 
                            data['roi_ix'].values)
    y = weights_to_vertices(np.ones(len(sig_regions)), Config().atlas, 
                            sig_regions['roi_ix'].values)
    z = weights_to_vertices(seed.values, Config().atlas, data['roi_ix'].values)
    seed_cmap = seed_color

    surfs = get_surfaces()
    sulc = get_sulc()
    if show_left_vis:
        p = Plot(surfs['rh'], views='posterior')
        sulc, x, y, z = [i[:int(len(i) / 2)] for i in [sulc, x, y, z]]
        p.add_layer(data=sulc, cmap='gray', cbar=False)
        #p.add_layer(np.nan_to_num(z), cmap=seed_cmap, cbar=False)
    else:    
        p = Plot(surfs['lh'], surfs['rh'])
        p.add_layer(data=sulc, cmap='gray', cbar=False)

    vmax = np.nanmax(np.abs(x))
    cmap = 'seismic'
    if sig_style == 'trace':
        #p.add_layer(x, cmap=cmap, color_range=(-vmax, vmax))
        p.add_layer(x, cmap=cmap, color_range=(-6, 6))
        p.add_layer(np.nan_to_num(y), as_outline=True, cmap='binary', cbar=False)
    elif sig_style == 'threshold':
        p.add_layer(x*np.nan_to_num(y), cmap=cmap, color_range=(-6, 6))
        #p.add_layer(x*np.nan_to_num(y), cmap=cmap, color_range=(-vmax, vmax), cbar=False)
    elif sig_style is None:
        #p.add_layer(x, cmap=cmap, color_range=(-vmax, vmax))
        p.add_layer(x, cmap=cmap, color_range=(-6, 6))

    p.add_layer(np.nan_to_num(z), cmap=seed_cmap, cbar=False)
    p.add_layer(np.nan_to_num(z), as_outline=True, cmap='binary', cbar=False)
    
    cbar_kws = dict(location='bottom', decimals=1, fontsize=14, n_ticks=3, shrink=.2, aspect=6, draw_border=False, pad=-.06)
    fig = p.build(cbar_kws=cbar_kws)

    return fig


def main():
    config = Config()
    cmats = get_files("/Users/keannarowchan/Desktop/statistical-manifolds/data/connectivity/sub*/sub*.tsv")
    ttest_data = pd.read_table("/Users/keannarowchan/Desktop/statistical-manifolds/data/ecc_ttest_stats.tsv")

    fig_dir = "/Users/keannarowchan/Desktop/statistical-manifolds/figures/seed/"
    os.makedirs(fig_dir, exist_ok=True)

    network_data = []
    seeds = []
    #clusters = [1, 2, 4, 1]
    #names = ['PMC', 'SPC', 'PMd', 'Vis']
    clusters = [1]
    names = ['Vis']    

   
    for clust, name in zip(clusters, names):

        if name == 'Vis':
            df = ttest_data.query("A == 'early' and B == 'late'")
        else:
            df = ttest_data.query("A == 'base' and B == 'early'")
        
        epochs = [['early', 'base'], ['late', 'early'], ['late', 'base']]
        
        for e in epochs:
            suffix = f'{e[1]}_vs_{e[0]}'
            res, seed, networks = seed_analysis(df, clust, cmats, e)
            networks['cluster'] = name
            network_data.append(networks)

            fig = plot_seed_map(res, seed['roi'], sig_style=None)
            out = os.path.join(fig_dir, f'{name}_{suffix}.png')
            fig.savefig(out, dpi=300)

            if name == 'Vis':
                fig = plot_seed_map(res, seed['roi'], sig_style=None, show_left_vis=True)
                out = os.path.join(fig_dir, f'{name}_{suffix}_post.png')
                fig.savefig(out, dpi=300)
        print(seed)
        seeds.append(seed)
    print(seeds)

main()
