import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.connectome import ConnectivityMeasure
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import sqrtm, logm, expm, invsqrtm

from config import Config
from utils import get_files, display

pjoin = os.path.join

def concat_timeseries_data(timeseries):
    config = Config()
    out_dir = pjoin(config.dataset_dir, config.tseries)
    os.makedirs(out_dir, exist_ok=True)
    
    subid = ['sub-01', 'sub-02', 'sub-03', 'sub-05', 'sub-06',  'sub-07',  'sub-10',  'sub-11',  'sub-12',  'sub-13',   'sub-14',   'sub-15',   'sub-16',  'sub-17',  'sub-18',  'sub-19',  'sub-20',  'sub-21',  'sub-22',  'sub-23',  'sub-24',  'sub-25',  'sub-26',  'sub-27',  'sub-28',  'sub-29',  'sub-30',  'sub-31',  'sub-32',  'sub-33',  'sub-34',  'sub-35',  'sub-36',  'sub-37']
    
    for sub in subid:
        appended_data = []
        all_timeseries_data = get_files(timeseries + sub + '*timeseries.tsv')
        for i in all_timeseries_data:
            fnames = os.path.split(i)[1]
            timeseries_datasets = pd.read_table(i)
            appended_data.append(timeseries_datasets)
        appended_data = pd.concat(appended_data)
        appended_data.to_csv(pjoin(out_dir, f'{sub}_stat-full-task_space-fsLR_den-91k_bold_timeseries.tsv'), sep='\t')

def _baseline_window_length():
    return 203

def do_not_drop_nontask_samps(x):
    return x

def _split_by_earlyandlate_learning(x):
    window_length = _baseline_window_length()

    baseline = x[:window_length]
    early = x[window_length: window_length*2]
    late = x[window_length*6: window_length*7]
    return [baseline, early, late]


def _split_by_learning(x):
    window_length = _baseline_window_length()

    baseline = x[:window_length]
    learning = x[window_length:]
    return [baseline, learning]

def compute_connectivity(timeseries, output_dir, float_fmt='%1.8f', split='earlyandlate'):
    
    fname = os.path.split(timeseries)[1]
    display(fname)
    data = pd.read_table(timeseries)
    tseries = data.values
    regions = data.columns

    # handle different scan types
    if 'full' in fname:
        tseries = do_not_drop_nontask_samps(tseries)
        
        if split == 'learning':
            dataset = _split_by_learning(tseries)
        elif split == 'earlyandlate':
            dataset = _split_by_earlyandlate_learning(tseries)
        else:
            dataset = [tseries]

    elif 'washout' in fname:
        dataset = [do_not_drop_nontask_samps(tseries)]
    else:
        dataset = [tseries]

    conn = ConnectivityMeasure(kind='covariance')
    connectivities = conn.fit_transform(dataset)

    output_dir = pjoin(output_dir, fname[:6])
    os.makedirs(output_dir, exist_ok=True)
    
    cmat_name = fname.split('_space')[0] + '_cmat'
    n_matrices = len(connectivities)
    if n_matrices == 2:
        suffix = ['_base', '_learn']
    elif n_matrices == 3:
        suffix = ['_base', '_early', '_late']
    else:
        suffix = ['']
        
    for s, cmat in zip(suffix, connectivities):
        out = pjoin(output_dir, cmat_name + s + '.tsv')
        out_cmat = pd.DataFrame(cmat, index=regions, columns=regions)
        out_cmat = out_cmat.drop(['Unnamed: 0'], axis=1)
        out_cmat = out_cmat.drop(['Unnamed: 0'], axis=0)
        out_cmat.to_csv(out, float_format=float_fmt, sep='\t')


def connectivity_analysis(input_data, out_dir, njobs=45):
    os.makedirs(out_dir, exist_ok=True)
    timeseries = get_files(input_data + '/*full*.tsv')
    
    for ts in timeseries:
        compute_connectivity(ts, out_dir)

def _to_tangent(s, mean):
    p = sqrtm(mean)
    p_inv = invsqrtm(mean)
    return p @ logm(p_inv @ s @ p_inv) @ p 
    

def _gl_transport(t, sub_mean, grand_mean):
    g = sqrtm(grand_mean) @ invsqrtm(sub_mean)
    return g @ t @ g.T


def _from_tangent(t, grand_mean):
    p = sqrtm(grand_mean)
    p_inv = invsqrtm(grand_mean)
    return p @ expm(p_inv @ t @ p_inv) @ p 


def center_cmat(c, sub_mean, grand_mean):
    t = _to_tangent(c, sub_mean)
    tc = _gl_transport(t, sub_mean, grand_mean)
    return _from_tangent(tc, grand_mean)


def center_subject(sub_cmats, grand_mean):
    sub_mean = mean_riemann(sub_cmats)
    return np.array([center_cmat(c, sub_mean, grand_mean) for c in sub_cmats])


def _read_and_stack_cmats(x):
    arr = np.array([pd.read_table(i, index_col=0).values for i in x])
    labels = pd.read_table(x[0], index_col=0).columns
    return arr, labels


def center_matrices(dataset_dir, float_fmt='%1.8f'):
    cmats = get_files([dataset_dir, '*/*.tsv'])
    out_dir = dataset_dir + '-centered'
    os.makedirs(out_dir, exist_ok=True)
    
    all_cmats, roi_labels = _read_and_stack_cmats(cmats)
    display('Computing grand mean')
    grand_mean = mean_riemann(all_cmats)

    # save grand mean
    df = pd.DataFrame(grand_mean, index=roi_labels, columns=roi_labels) 
    df.to_csv(pjoin(out_dir, 'grand_mean.tsv'), sep='\t', float_format=float_fmt)

    subs = np.unique([os.path.basename(x)[:6] for x in cmats])
    for s in subs:
        display(s)
        sub_files = [i for i in cmats if s in i]
        sub_cmats, _ = _read_and_stack_cmats(sub_files)
        centered_cmats = center_subject(sub_cmats, grand_mean)

        for i, fname in enumerate(sub_files):
            df = pd.DataFrame(centered_cmats[i], index=roi_labels, columns=roi_labels) 
            out_name = fname.replace(dataset_dir, out_dir)
            os.makedirs(os.path.split(out_name)[0], exist_ok=True)
            df.to_csv(out_name, sep='\t', float_format=float_fmt)

def main():
    config = Config()
    ts = pjoin(config.tseries + '/')
    concat_timeseries_data(ts)
    connectivity_analysis(config.tseries, config.connect)
    center_matrices(config.connect)

main()

