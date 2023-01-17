import os
from matplotlib import colors
import glob
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import natsort
from matplotlib.colors import LinearSegmentedColormap
import bct
from neuromaps import images
import glob
import subprocess
from datetime import datetime
import pandas as pd
import pingouin as pg
import nibabel as nib
from brainspace.mesh.mesh_io import read_surface
from surfplot import Plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from nilearn.plotting import cm
import cmasher as cmr
from brainspace.utils.parcellation import map_to_labels
from surfplot import Plot
from surfplot.utils import add_fslr_medial_wall
pjoin = os.path.join


class Config(object):
    resources = "/Users/keannarowchan/Desktop/statistical-manifolds/resources"
    atlas = "/Users/keannarowchan/Desktop/statistical-manifolds/atlases/Schaefer2018_1000Parcels_7Networks_order.dlabel.nii"
    adjacency = "/Users/keannarowchan/Desktop/statistical-manifolds/atlases/Schaefer2018_1000Parcels_7Networks_adjacency.tsv"
    gradients = "/Users/keannarowchan/Desktop/statistical-manifolds/data/pca-gradients-centered/"
    figures = "/Users/keannarowchan/Desktop/statistical-manifolds/figures"
    k = 3
    
    os.makedirs(gradients, exist_ok=True)
    os.makedirs(figures, exist_ok=True)

def parse_roi_names(x, col='roi'):
    roi_cols = ['hemi', 'network', 'name']
    x[roi_cols] = x[col].str.split('_', n=3, expand=True).iloc[:, 1:]
    return x

def get_clusters(data, adjacency, sort=True, yuh=False):

    ix = data.query("sig_corrected == 1")['roi_ix'].values
    
    adjacency.columns = adjacency.columns.astype(int)
    x = adjacency.loc[ix, ix].values
    assignments, sizes = bct.get_components(x)

    cluster_table = pd.DataFrame({
        'cluster': np.arange(len(sizes)) + 1, 
        'size': sizes
    })

    res = data.copy()
    res.index = res['roi_ix'].values
    res['cluster'] = 0
    res.loc[ix, 'cluster'] = assignments
    res = res.merge(cluster_table, on='cluster', how='left')
    res['size'] = np.nan_to_num(res['size'])

    if sort:
        labels = res.sort_values('size', ascending=False)['cluster'].unique()
        new_labels = np.concatenate([np.arange(len(labels[:-1])) + 1, [0]])
        relabel_map = dict(zip(labels, new_labels))
        res['cluster'] = res['cluster'].apply(lambda x: relabel_map[x])

    return res

def get_surfaces(style='inflated', load=True):
    config = Config()
    surf_path = os.path.join(config.resources, 'surfaces')
    surfaces = get_files([surf_path, f'*.{style}_*'])
    
    if load:
        surfs = [read_surface(i) for i in surfaces]
        return dict(zip(['lh', 'rh'], surfs))
    else:
        return surfaces


def test_regions(data, method='anova', p_thresh=.05):

    test = dict(anova=pg.rm_anova, ttest=pg.pairwise_ttests)
    if method not in test.keys():
        raise ValueError(f"method must be one of {list(test.keys())}")
    
    test_data = data[['sub', 'roi', 'roi_ix', 'epoch', 'distance']]

    kwargs = dict(correction=True) if method == 'anova' else {}
    res = test_data.groupby(['roi', 'roi_ix'], sort=False) \
                   .apply(test[method], dv='distance', within='epoch', 
                          subject='sub', **kwargs) \
                   .reset_index() \
                   .drop('level_2', axis=1)
    res['sig'] = (res['p-unc'] < p_thresh).astype(float)
    res = fdr_correct(res)
    
    adj = pd.read_table(Config().adjacency, index_col=0)
    if method == 'ttest':
        # raise Exception 
        res = res.groupby(['A', 'B'], sort=False) \
                 .apply(get_clusters, adj) \
                 .reset_index(drop=True)
    else:
        res = get_clusters(res, adj)
    return res

def fdr_correct(x, colname='p-unc'):
    """Apply FDR correction across all rows of dataframe"""
    corrected = pg.multicomp(x[colname].values, method='fdr_bh')
    x[['sig_corrected', 'p_fdr']] = np.array(corrected).T
    return 

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

    # configure scale
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

def stat_cmap():
    return cmr.get_sub_cmap(cm.cyan_orange, .05, .95)

def get_sulc():
    config = Config()
    surf_path = os.path.join(config.resources, 'surfaces')
    img = os.path.join(surf_path, 'S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii')
    vertices = nib.load(img).get_fdata().ravel()
    return add_fslr_medial_wall(vertices)


def pairwise_stat_maps(data, prefix, dorsal=True, posterior=True, vmax='auto', vmin='auto', cbar_orientation='vertical'):
    pairwise_contrasts = data.groupby(['A', 'B'])
    list_ = []
    for name, g in pairwise_contrasts:
        contrast = f'{name[0]}_{name[1]}'
        # get stats of only significant regions
        sig = g[g['sig_corrected'].astype(bool)]
        sig.set_index('roi_ix', inplace=True)
        sig = sig[['T']]
        sig = sig.rename(columns={'T': contrast + '_T'})
        list_.append(sig)
    df = pd.concat(list_, axis=1)

    tvals = -df.filter(like='T')
    
    if vmax == 'auto':
        vmax = np.nanmax(tvals.values)
        
    if vmin == 'auto':
        vmin = np.nanmin(np.abs(tvals.values))

    # draw separate pos/neg cmaps
    if cbar_orientation == 'vertical':
        size = (.2, 1)
    else:
        size = (.8, .25)

    cmap = stat_cmap()
    pos_cmap = cmr.get_sub_cmap(cmap, .51, 1)
    pos_cmap = cmr.get_sub_cmap(pos_cmap, vmin/vmax, 1)
    plot_cbar(pos_cmap, vmin, vmax, cbar_orientation, size=size, n_ticks=2)
    plt.savefig(prefix + 'cbar_pos')

    neg_cmap = cmr.get_sub_cmap(cmap, 0, .5)
    neg_cmap = cmr.get_sub_cmap(neg_cmap, 0, 1 - vmin/vmax)
    plot_cbar(neg_cmap, -vmax, -vmin, cbar_orientation, size=size, n_ticks=2)
    plt.savefig(prefix + 'cbar_neg')

    config = Config()
    surfaces = get_surfaces()
    sulc = get_sulc()
    # cmap = cmr.get_sub_cmap(cm.cyan_orange, .15, .83)
    cmap = stat_cmap()
    sulc_params = dict(data=sulc, cmap='gray', cbar=False)
    layer_params = dict(cmap=cmap, cbar=False, color_range=(-vmax, vmax))
    outline_params = dict(cbar=False, cmap='binary', as_outline=True)

    for i in tvals.columns:
        contrast = i[:-2].replace('_', '_vs_')
        x = weights_to_vertices(tvals[i], config.atlas, tvals.index.values)

        p = Plot(surfaces['lh'], surfaces['rh'], layout='row', mirror_views=True, size=(800, 200), zoom=1.2)
        p.add_layer(**sulc_params)
        p.add_layer(x, **layer_params)
        p.add_layer((np.nan_to_num(x) != 0).astype(float), **outline_params)
        fig = p.build(colorbar=False)
        fig.savefig(prefix + contrast)

        if dorsal:
            p = Plot(surfaces['lh'], surfaces['rh'], views='dorsal', size=(150, 200), zoom=3.3)
            p.add_layer(**sulc_params)
            p.add_layer(x, **layer_params)
            p.add_layer((np.nan_to_num(x) != 0).astype(float), **outline_params)
            fig = p.build(colorbar=False)
            fig.savefig(prefix + contrast + '_dorsal')

        if posterior:
            p = Plot(surfaces['lh'], surfaces['rh'], views='posterior', size=(150, 200), zoom=3.3)
            p.add_layer(**sulc_params)
            p.add_layer(x, **layer_params)
            p.add_layer((np.nan_to_num(x) != 0).astype(float), **outline_params)
            fig = p.build(colorbar=False)
            fig.savefig(prefix + contrast + '_posterior')

    return fig

def eccentricity_analysis(data):
    anova = test_regions(data)

    sig_regions = anova.loc[anova['sig_corrected'].astype(bool), 'roi'].tolist()
    if sig_regions:
        post_data = data[data['roi'].isin(sig_regions)]
        posthoc = test_regions(post_data, 'ttest')  
    else:
        posthoc = None

    return anova, posthoc


def mean_stat_map(data, out_dir, centering='baseline'):
    mean_ecc = data.groupby(['epoch', 'roi', 'roi_ix'])['distance'] \
                   .mean() \
                   .reset_index() \
                   .sort_values(['epoch', 'roi_ix'])
    
    # roi by epoch (in chronological order) data for plotting
    epoch_ecc = pd.DataFrame({name: g['distance'].values for name, g in mean_ecc.groupby('epoch')})
    
    if centering is not None:
        if centering == 'mean':
            epoch_ecc = epoch_ecc.apply(lambda x: x - x.mean(), axis=1)
            prefix = 'centered_mean_ecc_'
        elif centering == 'baseline':
            epoch_ecc['early'] -= epoch_ecc['base']
            epoch_ecc['late'] -= epoch_ecc['base']
            epoch_ecc.drop('base', axis=1, inplace=True)
            prefix = 'relative_mean_ecc_'

        cmap = "summer"
        vmax = np.around(np.nanmax(epoch_ecc), 2)
        vmin = -vmax
        n_ticks = 3
    else:
        cmap = 'summer'
        vmax = np.nanmax(epoch_ecc)
        vmin = np.nanmin(epoch_ecc)
        n_ticks = 2
        prefix = 'mean_ecc_'

    prefix = os.path.join(out_dir, prefix)
    plot_cbar(cmap, vmin, vmax, 'vertical', size=(.2, 1), n_ticks=n_ticks)
    plt.savefig(prefix + 'cbar') 

    config = Config()
    surfaces = get_surfaces()

    for i in epoch_ecc.columns:
        x = weights_to_vertices(epoch_ecc[i], config.atlas)

        p = Plot(surfaces['lh'], surfaces['rh'], layout='row', mirror_views=True, size=(800, 200), zoom=1.2)
        p.add_layer(x, cmap=cmap, color_range=(vmin, vmax), cbar=False)
        fig = p.build()
        fig.savefig(prefix + i + '_brain')

    return fig

def plot_mean_scatters(data, out_dir, view_3d=(30, -110), eccentricity=False):
    k = [f'g{i}' for i in np.arange(3) + 1]
    mean_loadings = data.groupby(['epoch', 'roi'])[k + ['distance']] \
                        .mean() \
                        .reset_index()
    mean_loadings = parse_roi_names(mean_loadings)

    if eccentricity:
        c_col = 'distance'
        cmap = 'viridis'
        vmax = np.nanmax(mean_loadings['distance'])
        vmin = np.nanmin(mean_loadings['distance'])
        suffix = 'scatter_ecc'
    else:
        c_col='c'
        cmap = yeo_cmap()
        mean_loadings['c'] = mean_loadings['network'].apply(lambda x: cmap[x])
        vmax, vmin = None, None
        suffix = 'scatter'

    for epoch in ['base', 'early', 'late']:
        df = mean_loadings.query("epoch == @epoch")

        x, y, z = k
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax = plot_3d(df[x], df[y], df[z], color=df[c_col], s=10, lw=0, ax=ax, view_3d=view_3d, vmax=vmax, vmin=vmin)
        ax.set(xlim=(-3.5, 5), ylim=(-3.5, 5), zlim=(-3.5, 4))
        
        if epoch == 'base':
            title = 'Baseline'
        else:
            title = epoch
        ax.set_title(title.title())
        
        prefix = os.path.join(out_dir, f'mean_{epoch}_')
        fig.savefig(prefix + suffix)

def anova_stat_map(anova, out_dir, outline=True):
    df = anova.query("sig_corrected == 1")
    fvals = df['F'].values
    vmax = int(np.nanmax(fvals)) 
    vmin = np.nanmin(fvals)

    cmap = "autumn"
    cmap_min = vmin / vmax

    plot_cbar(cmap, vmin, vmax, 'horizontal', size=(1, .3), n_ticks=2)
    prefix = os.path.join(out_dir, 'anova')
    plt.savefig(prefix + '_cbar')

    surfaces = get_surfaces()
    sulc = get_sulc()
    x = pweights_to_vertices(fvals, Config().atlas, df['roi_ix'].values)
    sulc_params = dict(data=sulc, cmap='gray', cbar=False)
    layer_params = dict(cmap=cmap, cbar=False, color_range=(vmin, vmax))
    outline_params = dict(data=(np.abs(x) > 0).astype(bool), cmap='binary', cbar=False, as_outline=True)

    # 2x2 grid
    p = Plot(surfaces['lh'], surfaces['rh'])
    p.add_layer(**sulc_params)
    p.add_layer(x, **layer_params)
    if outline:
        p.add_layer(**outline_params)

    cbar_kws = dict(n_ticks=2, aspect=8, shrink=.15, draw_border=False)
    fig = p.build(cbar_kws=cbar_kws)
    fig.savefig(prefix)

    # dorsal views
    p = Plot(surfaces['lh'], surfaces['rh'], views='dorsal', size=(150, 200), zoom=3.3)
    p.add_layer(**sulc_params)
    p.add_layer(x, **layer_params)
    if outline:
        p.add_layer(**outline_params)
    fig = p.build(colorbar=False)
    fig.savefig(prefix + '_dorsal')

    # posterior views
    p = Plot(surfaces['lh'], surfaces['rh'], views='posterior', size=(150, 200), zoom=3.3)
    p.add_layer(**sulc_params)
    p.add_layer(x, **layer_params)
    if outline:
        p.add_layer(**outline_params)
    fig = p.build(colorbar=False)
    fig.savefig(prefix + '_posterior')

    return x

def _ensemble_cmap(as_cmap=False):

    colors = ['tab:cyan', 'orangered', 'tab:purple', 'tab:olive']
    if as_cmap:
        return LinearSegmentedColormap.from_list('cmap', colors, N=4)
    else:
        return dict(zip(range(1, 5), colors))

def ensemble_analysis(gradients, anova, out_dir, k=3):
    cols = [f'g{i}' for i in np.arange(k) + 1]
    base_loadings = gradients.query("epoch == 'base'")

    sig_rois = anova.query("sig_corrected == 1")['roi'].tolist()
    df = base_loadings.query("roi in @sig_rois") \
                      .groupby(['epoch', 'roi'], sort=False)[cols] \
                      .mean() \
                      .reset_index()
    kmeans = KMeans(n_clusters=4, random_state=1234)
    kmeans.fit(df[cols])
    df['ensemble'] = kmeans.labels_ + 1

    res = base_loadings.merge(df[['roi', 'ensemble']], on='roi', how='left')

    # brain plot
    x = weights_to_vertices(res['ensemble'].values, Config().atlas)
    x = np.nan_to_num(x)
    surfaces = get_surfaces()
    sulc = get_sulc()
    cmap = _ensemble_cmap(True)

    p = Plot(surfaces['lh'], surfaces['rh'])
    p.add_layer(data=sulc, cmap='gray', cbar=False)
    p.add_layer(x, cbar=None, cmap=cmap)
    p.add_layer((np.abs(x) > 0).astype(bool), as_outline=True, cbar=None, cmap='binary')
    fig = p.build()
    fig.savefig(os.path.join(out_dir, 'anova_ensembles.png'), dpi=300)

    data = gradients.merge(df[['roi', 'ensemble']], on='roi', how='left')
    data = data.groupby(['sub', 'epoch', 'ensemble'])['distance'] \
               .mean() \
               .reset_index()

    colors = list(_ensemble_cmap().values())
    g = sns.FacetGrid(data=data, col_wrap=2, col='ensemble', hue='ensemble', palette=colors, height=1.8)
    g.map_dataframe(sns.lineplot, x='epoch', y='distance', ci=None, marker='o', ms=5, lw=1.2, mfc='w', mec='k', color='k')
    g.map_dataframe(sns.stripplot, x='epoch', y='distance', jitter=.1, zorder=-1, s=4, alpha=.752)
    g.set_axis_labels('', "Eccentricity")
    g.set(xticklabels=['Baseline', 'Early', 'Late'])
    g.set_titles('')
    g.savefig(os.path.join(out_dir, 'ensemble_ecc_plot.png'), dpi=300)
    return df[['roi', 'ensemble']]


def plot_displacements(data, anova, k=3, ax=None, hue='network'):
    if isinstance(k, int):
        k = [f'g{i}' for i in np.arange(k) + 1]

    ensb = data[['roi', 'ensemble']].groupby('roi', sort=False).first()

    mean_loadings = data.groupby(['epoch', 'roi'])[k].mean().reset_index()
    mean_loadings = parse_roi_names(mean_loadings)
    mean_loadings = mean_loadings.merge(ensb, on='roi', how='left')

    base = mean_loadings.query("epoch == 'base'")
    sig_regions = anova.loc[anova['sig_corrected'].astype(bool), 'roi']
    sig_base = base[base['roi'].isin(sig_regions)]
    shifts = mean_loadings[mean_loadings['roi'].isin(sig_regions)]

    if hue == 'network':
        cmap = yeo_cmap()
    elif hue == 'ensemble':
        cmap = _ensemble_cmap()
    if len(k) == 2:
        x, y = k
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        
        # all regions  
        sns.scatterplot(x=x, y=y, data=base, color='k', alpha=.3, s=5, linewidths=0, legend=False, ax=ax)

        # plot shifts/lines of significant regions
        for roi in shifts['roi'].unique():
            roi_df = shifts.query("roi == @roi")
            xx = roi_df[x].values
            yy = roi_df[y].values
            val = roi_df[hue].iloc[0]
            ax.plot(xx, yy, lw=1, c=cmap[val])
            
            arrowprops = dict(lw=.1, width=.1, headwidth=4, headlength=3, color=cmap[val])
            ax.annotate(text='', xy=(xx[-1], yy[-1]), xytext=(xx[-2], yy[-2]), arrowprops=arrowprops)
        
        sns.scatterplot(x=x, y=y, data=sig_base, hue=hue, s=16, edgecolor='k', palette=cmap, linewidths=1, ax=ax, legend=False, zorder=20)
        sns.despine()
        return ax
    
    elif len(k) == 3:
        x, y, z = k
        fig = plt.figure(figsize=(8, 4))
        gs = fig.add_gridspec(nrows=10, ncols=10)
        ax1 = fig.add_subplot(gs[:, :6], projection='3d')

        base_nonsig = base[~base['roi'].isin(sig_regions)]
        ax1 = plot_3d(base_nonsig[x], base_nonsig[y], base_nonsig[z], color='gray', alpha=.3, s=1, ax=ax1, view_3d=(35, -110))
        ax1.set(xticks=range(-4, 6))

        for roi in shifts['roi'].unique():
            roi_df = shifts.query("roi == @roi")
            xx = roi_df[x].values
            yy = roi_df[y].values
            zz = roi_df[z].values
            val = roi_df[hue].iloc[0]
            ax1.plot(xs=xx, ys=yy, zs=zz, lw=1, c=cmap[val])
        
        sig_base['c'] = sig_base[hue].apply(lambda x: cmap[x])
        ax1 = plot_3d(sig_base[x], sig_base[y], sig_base[z], color=sig_base['c'], alpha=1, s=20, ax=ax1, zorder=20, edgecolors='k', linewidths=.5)

        ax2 = fig.add_subplot(gs[:5, 6:9])
        ax2 = plot_displacements(data, anova, ['g1', 'g2'], ax=ax2)
        ax2.set(ylim=(-4, 5), xlim=(-4, 5), xticklabels=[], xlabel='', ylabel='PC2')
        ax3 = fig.add_subplot(gs[5:, 6:9])
        ax3 = plot_displacements(data, anova, ['g1', 'g3'], ax=ax3)
        ax3.set(ylim=(-4, 5), xlim=(-4, 5), xticks=np.arange(-4, 6, 2), xlabel='PC1', ylabel='PC3')

        fig.tight_layout()
        return fig, ax
    else:
        return None, None

def main():

    config = Config()
    fig_dir = "/Users/keannarowchan/Desktop/statistical-manifolds/figures/adaptation/"

    gradients = pd.read_table("/Users/keannarowchan/Desktop/statistical-manifolds/data/subject_gradients.tsv")
    anova_stats, ttest_stats = eccentricity_analysis(gradients)

    #anova_stats.to_csv(os.path.join(config.results, 'ecc_anova_stats.tsv'), sep='\t', index=False)
    #ttest_stats.to_csv(os.path.join(config.results, 'ecc_ttest_stats.tsv'), sep='\t', index=False)


    mean_stat_map(gradients, fig_dir)
    mean_stat_map(gradients, fig_dir, 'mean')
    mean_stat_map(gradients, fig_dir, 'baseline')
    
    #anova_vertices = anova_stat_map(anova_stats, fig_dir)
    #np.savetxt(os.path.join(config.results, 'anova_vertices.tsv'), anova_vertices)
    #plotting.pairwise_stat_maps(ttest_stats, os.path.join(fig_dir, 'ecc_ttests_'))
    
    #ensb = ensemble_analysis(gradients, anova_stats, fig_dir, config.k)
    #gradients = gradients.merge(ensb, on='roi', how='left')
    #anova_stats = anova_stats.merge(ensb, on='roi', how='left')


    #plot_mean_scatters(gradients, fig_dir)
    #plot_mean_scatters(gradients, fig_dir, eccentricity=True)
    #fig, _ = plot_displacements(gradients, anova_stats, config.k)
    #fig.savefig(os.path.join(fig_dir, 'displacements'))

    #fig, _ = plot_displacements(gradients, anova_stats, config.k, hue='ensemble')
    #fig.savefig(os.path.join(fig_dir, 'displacements_by_ensemble'))

main()
