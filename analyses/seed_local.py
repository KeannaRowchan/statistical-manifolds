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
import cmasher as cmr
import bct
from neuromaps import images
import glob
import subprocess
from datetime import datetime
import natsort
import numpy as np
import pandas as pd
import pingouin as pg
import nibabel as nib
from brainspace.mesh.mesh_io import read_surface
from surfplot import Plot
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nilearn.plotting import cm
import cmasher as cmr
from brainspace.utils.parcellation import map_to_labels
from surfplot import Plot
from surfplot.utils import add_fslr_medial_wall
pjoin = os.path.join

class Config(object):

   #config_path = os.path.dirname(os.path.abspath(__file__))
    #resources = pjoin(config_path, '../resources')
    #fmriprep_dir = '/Raid6/raw/VMR-Learning-Complete/derivatives/2020/fmriprep'

    # parcellation
    #atlas = "Volumes/KeannaRowchan/statistical-manifolds/statistical-manifolds/resources/atlas/Schaefer2018_1000Parcels_7Networks_order.dlabel.nii"
    atlas = "/Users/keannarowchan/Desktop/server_to_desktop/Schaefer2018_1000Parcels_7Networks_order.dlabel.nii"
    #adjacency = pjoin(resources, 'atlases', 'Schaefer2018_1000Parcels_7Networks_adjacency.tsv')
    
    # data directories
    #data_dir = pjoin(config_path, '../data')
    #dataset = 'schaefer1000-7networks-final'
    #dataset_dir = "Users/keannarowchan/Desktop/server_to_desktop"
    #tseries = pjoin(dataset_dir, 'timeseries')
    #connect = pjoin(dataset_dir, 'connectivity')

    # paths expected to change
    #gradients = pjoin(dataset_dir, 'pca-gradients-centered-ses-01')

    k = 3
    #results = pjoin(config_path, f'../results/k{k}')
    #figures = pjoin(config_path, f'../figures/fig-components-pngs-k{k}')

    #os.makedirs(results, exist_ok=True)
    #os.makedirs(figures, exist_ok=True)

def _align_labels_to_atlas(x, source_labels, target_labels):
    """Match labels to corresponding vertex labels"""

    target = np.unique(target_labels)[1:]
    df1 = pd.DataFrame(target, index=target)
    df2 = pd.DataFrame(x, index=source_labels)
    return pd.concat([df1, df2], axis=1).iloc[:, 1:].values

def get_surfaces(style='inflated', load=True):
    """Fetch surface files of a given surface style
    Parameters
    ----------
    style : str, optional
        Type of surface to return, by default 'inflated'
    Returns
    -------
    dict
        Dictionary of left (lh) and right (rh) surface files
    """
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
    """Get sulcal depth map for plotting style"""
    config = Config()
    surf_path = "/Users/keannarowchan/Desktop/server_to_desktop/surfaces"
    img = os.path.join(surf_path, 'S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii')
    vertices = nib.load(img).get_fdata().ravel()
    return add_fslr_medial_wall(vertices)

def weights_to_vertices(data, target, labels=None):
    """Map weights (e.g., gradient loadings) to vertices on surface
    If `labels` is not specifiied, values in `data` are mapped to `target` in 
    ascending order.
    Parameters
    ----------
    data : numpy.ndarray or str
        Array containing region weights of shape (n_regions, n_features). If
        more than one feature/column is detected, then brain maps for each 
        feature are created. If a string, must be a valid CIFTI file name
    target : str
        CIFTI file name (dlabel or dscalar) that defines vertex-space for 
        mapping 
    labels : numpy.ndarray
        Numeric labels for each region (i.e. row of `data`) as they appear in 
        the atlas vertices. Required when `data` contains fewer regions than
        total regions in `target`, as is the case when `data` is a result of 
        some thresholded/indexing (e.g., `data` only contains weights of 
        significant regions). By default None.
    Returns
    -------
    numpy.ndarray
        Array of mapped vertices
    """
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
    # drop indices of missing regions in Schaefer 1000. These values/regions 
    # appear in the dlabel.nii labels, but not in the actual vertex array, as
    # they have been 'upsampled-out' of the atlas
    return x[~np.isin(x, [533, 903])]

def get_files(pattern, force_list=False):
    """Extracts files in alphanumerical order that match the provided glob 
    pattern.
    Parameters
    ----------
    pattern : str or list
        Glob pattern or a list of strings that will be joined together to form 
        a single glob pattern.  
    force_list : bool, optional
        Force output to be a list. If False (default), a string is returned in
        cases where only one file is detected.
    Returns
    -------
    str or list
        Detected file(s).
    Raises
    ------
    FileNotFoundError
        No files were detected using the input pattern.
    """
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
    """Identify cluster seeds using region with max or closest-to-mean T value
    in contrast
    Parameters
    ----------
    df : pandas.DataFrame
        Contrast data (e.g., Early vs Baseline)
    method : str, optional
        Selection approach, by default 'middle', which is the closest-to-mean
        region
    Returns
    -------
    str
        Seed region name
    """
    if method == 'max':
        return df.loc[df['T'].idxmax()]
    elif method == 'middle':
        return df.loc[np.abs(df['T'] - df['T'].mean()).idxmin()]


def connect_seed(cmats, seed_region):
    """Extract seed connectivity by isolating row in connectivity matrix
    Parameters
    ----------
    cmats : list
        Connectivity matrices
    seed_region : str
        Seed region name
    Returns
    -------
    pandas.DataFrame
        Region connectivity profiles across subjects
    """
    list_ = []
    for i in cmats:
        cmat = pd.read_table(i, index_col=0)
        # isolate row of seed region
        res = pd.DataFrame(cmat.loc[seed_region].reset_index().values, 
                           columns=['roi', 'r'])
        
        res['roi_ix'] = schaefer1000_roi_ix()

        fname = os.path.basename(i)
        res['sub'] = fname.split('_')[0]
        res['epoch'] = fname.split('_')[-1].split('.')[0]
        list_.append(res)

    connectivity = pd.concat(list_)
    connectivity['r'] = connectivity['r'].astype(np.float)
    return connectivity


def seed_analysis(contrasts, clust_num, cmats, epochs):
    """Perform seed connectivity contrast analysis
    Parameters
    ----------
    contrasts : pandas.DataFrame
        Eccentricity contrast results
    clust_num : int
        Eccentricity contrast cluster number
    cmats : List
        Connectivity matrices
    epochs : _type_
        Task epochs to compare connectivity, not necessarily the same task 
        epochs from the eccentricity contrast  
    Returns
    -------
    pandas.DataFrame, str, pandas.DataFrame
        Seed connectivity results (region and networks), and seed name
    """
    seed = find_cluster_seed(contrasts.query("cluster == @clust_num"))
    connectivity = connect_seed(cmats, seed['roi'])

    df = connectivity.query("epoch in @epochs")
    res = df.groupby(['roi', 'roi_ix'], sort=False) \
            .apply(pg.pairwise_ttests, dv='r', within='epoch', subject='sub') \
            .reset_index()
    # swap sign so that B condition is the positive condition
    res['T'] = -res['T']
    res['sig'] = (res['p-unc'] < .05).astype(float)

    # get network-level changes
    df = parse_roi_names(df)
    networks = df.groupby(['sub', 'network', 'epoch']) \
                 .mean() \
                 .reset_index()
    return fdr_correct(res), seed, networks


def plot_seed_map(data, seed_region, sig_style=None, use_fdr=True, seed_color="Greens", show_left_vis=False):
    """Generate seed connectivity contrast maps
    Parameters
    ----------
    data : pandas.DataFrame
        Region-wise seed connectivity results
    seed_region : str
        Seed region name
    sig_style : str, optional
        Significance indication, by default None
    use_fdr : bool, optional
        If showing significance, show FDR-corrected results, by default True
    seed_color : str, optional
        Seed region color, by default 'yellow'
    show_left_vis : bool, optional
        Show left visual cortex, which is necessary for the visual seed only, 
        by default False
    Returns
    -------
    matplotlib.figure.Figure
        Seed contrast stat map
    """
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
    #seed_cmap = colors.LinearSegmentedColormap.from_list('regions', [seed_color, 'k'], N=2)

    surfs = get_surfaces()
    sulc = get_sulc()
    if show_left_vis:
        p = Plot(surfs['rh'], views='posterior')
        sulc, x, y, z = [i[:int(len(i) / 2)] for i in [sulc, x, y, z]]
        p.add_layer(data=sulc, cmap='gray', cbar=False)
        #p.add_layer(np.nan_to_num(z), cmap=seed_cmap, cbar=False)
    else:    
        p = Plot(surfs['lh'], surfs['rh'], views='posterior')
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
    cmats = get_files("/Users/keannarowchan/Desktop/server_to_desktop/connectivity/sub*/sub*.tsv")
    ttest_data = pd.read_table("/Users/keannarowchan/Desktop/server_to_desktop/ecc_ttest_stats.tsv")

    fig_dir = "/Users/keannarowchan/Desktop/server_to_desktop/"
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
