import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from brainspace.gradient.utils import dominant_set
import bct
from neuromaps import stats

from surfplot import Plot

from config import Config
from utils import get_surfaces, permute_map

from plotting import set_plotting, weights_to_vertices

FIG_DIR = os.path.join(Config().figures, 'measures')
os.makedirs(FIG_DIR, exist_ok=True)
config = Config()

#plotting.set_plotting()

def compute_measures(cmat, thresh=.9):
    regions = cmat.index.str.split('_', n=3, expand=True).to_frame()
    network_aff = regions[2].values
    network_hemi_aff = (regions[1] + '_' + regions[2]).values

    x = dominant_set(cmat.values, k=1-thresh, is_thresh=False, as_sparse=False)
    x_bin = (x != 0).astype(float)
    
    measures = pd.DataFrame({
        'roi': cmat.index.values, 
        'hemi': regions[1].values,
        'network': network_aff,
        'participation': bct.participation_coef(x, network_aff, 'out'),
        'participation_h': bct.participation_coef(x, network_hemi_aff, 'out'),
        'module_degree': bct.module_degree_zscore(x_bin, network_aff, 2),
        'module_degree_h': bct.module_degree_zscore(x_bin, network_hemi_aff, 2),
        'strength': np.sum(x, axis=1)
    })
    return measures

def correlate_measures(data, out_dir, parc, n_perm=1000):
    y = data['distance'].values

    correlations = []
    distributions, spin_data = {}, {}
    measures = data.drop(['roi', 'hemi', 'network'], axis=1).columns
    for i in measures:
        x = data[i].values
        spins = permute_map(x, parc, n_perm=n_perm)

        r = stats.efficient_pearsonr(x, y, nan_policy='omit')[0]
        nulls = stats.efficient_pearsonr(y, spins, nan_policy='omit')[0]
        p = (np.sum(np.abs(nulls) >= np.abs(r)) + 1) / (len(nulls) + 1)

        correlations.append({'measure': i, 'r': r, 'p': p})
        distributions[i] = nulls
        spin_data[i] = spins

    correlations = pd.DataFrame(correlations)
    distributions = pd.DataFrame(distributions)
    
    for name, df in zip(['corrs', 'nulls'], [correlations, distributions]):
        df.to_csv(os.path.join(out_dir, f'ecc_fc_{name}.tsv'), sep='\t', index=False)

    for k, v in spin_data.items():
        np.savetxt(os.path.join(out_dir, f'{k}_spins.tsv'), v, delimiter='\t')

    return correlations, distributions, spin_data

def run_correlations(data, out_dir, overwrite=True):
    if overwrite or (not os.path.exists(out_dir)):
        os.makedirs(out_dir, exist_ok=True)
        corrs, nulls, spins = correlate_measures(data, out_dir, parc=config.atlas)
    else:
        corrs = pd.read_table(os.path.join(out_dir, 'ecc_fc_corrs.tsv'))
        nulls = pd.read_table(os.path.join(out_dir, 'ecc_fc_nulls.tsv'))
        
        spins = {}
        for i in os.listdir(out_dir):
            if 'spins' in i:
                name = i[:-10]
                spins[name] = np.loadtxt(os.path.join(out_dir, i))
    return corrs, nulls, spins

def _plot_measure_map(data, atlas, cmap='viridis', color_range=None, 
                      cbar_kws=None):
    surfs = get_surfaces()
    x = plotting.weights_to_vertices(data, config.atlas)

    p = Plot(surfs['lh'], surfs['rh'])
    p.add_layer(x, cmap=cmap, zero_transparent=False, color_range=color_range)
    fig = p.build(cbar_kws=cbar_kws)
    return fig

def plot_maps(data, out_dir):
    # node strength
    vmax = data['strength'].max()
    cbar_kws = dict(location='right', n_ticks=2, aspect=7, shrink=.15, draw_border=False, pad=-.05)

    fig = _plot_measure_map(data['strength'], config.atlas, cbar_kws=cbar_kws)
    fig.axes[0].set_title('Node strength', fontsize=14)
    fig.savefig(os.path.join(out_dir, 'strength_map'))

    # participation coefficent
    vmax = data['participation'].max()
    cbar_kws = dict(location='right', n_ticks=2, aspect=7, shrink=.15, draw_border=False, pad=-.05)

    fig = _plot_measure_map(data['participation'], atlas, cbar_kws=cbar_kws, color_range=(0, data['participation'].max()))
    fig.axes[0].set_title('Participation\ncoefficient', fontsize=14)
    fig.savefig(os.path.join(out_dir, 'participation_map'))

    # within-module degree zscore
    vmax = data['module_degree'].max()
    cbar_kws = dict(location='right', n_ticks=3, aspect=7, shrink=.15, draw_border=False, pad=-.05)

    fig = _plot_measure_map(data['module_degree'], atlas, color_range=(-vmax, vmax), cbar_kws=cbar_kws)
    fig.axes[0].set_title('Within-module\ndegree zscore', fontsize=14)
    fig.savefig(os.path.join(out_dir, 'module_degree_map'))
    
def plot_correlations(data, out_dir):
    measures = ['strength', 'participation', 'module_degree']
    xlabels = ['Node strength\n', 'Participation\ncoefficient','Within-module\ndegree zscore']
    for i, m in enumerate(measures):

        scatter_kws = dict(s=4, alpha=1, linewidth=0, color='tab:gray', clip_on=False)
        line_kws = dict(color='k', lw=1)
        fig, ax = plt.subplots(figsize=(2, 2))
        ax = sns.regplot(x=m, y='distance', data=data, scatter_kws=scatter_kws, line_kws=line_kws, ax=ax, ci=None)
        
        ylim = (np.floor(data['distance'].min()), np.ceil(data['distance'].max()))
        xlim = (np.around(data[m].min(), 3), np.around(data[m].max(), 3))
        ax.set(xlabel=xlabels[i], ylabel='Eccentricity', ylim=ylim, xlim=xlim)
        ax.set_yticks(np.arange(0, 6))
        ax.set_ylim(0, 5)
        ticks = [tick for tick in ax.get_xticks()]
        ax.set_xlim(ticks[0], ticks[-1])
        sns.despine()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{m}_scatter'))
        
def main():
    config = Config()

    fname = os.path.join(config.connect + '-centered','reference_cmat.tsv')
    cmat = pd.read_table(fname, index_col=0)
    data = compute_measures(cmat)
    
    ecc = pd.read_table(os.path.join(config.results, 'ref_ecc.tsv'), sep='\t')
    data = data.merge(ecc, on='roi')

    out_dir = os.path.join(config.results, 'fc_correlations')
    corrs, nulls, spins = run_correlations(data, out_dir)

    #plot_maps(data, FIG_DIR)
    plot_correlations(data, FIG_DIR)

main()
