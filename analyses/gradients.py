import os
import re
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps
from pyriemann.utils.mean import mean_riemann

from config import Config
from utils import get_files, display

def _pca_gradients(x, ref=None):
    alignment = None if ref is None else 'procrustes'
    gm = GradientMaps(n_components=None, approach='pca', kernel='cosine', alignment=alignment)
    gm.fit(x, reference=ref)
    return gm


def _dm_gradients(x, ref=None):
    alignment = None if ref is None else 'procrustes'
    gm = GradientMaps(n_components=20, approach='dm', kernel='cosine', alignment=alignment, random_state=42)
    gm.fit(x, diffusion_time=0, reference=ref)
    return gm


def save_gradients(gradients, lambdas, regions, out_prefix, float_fmt=None, save_threshold=.01):
    n_gradients = gradients.shape[1]
    labels = [f'g{x}' for x in np.arange(n_gradients) + 1]
    
    gradients = pd.DataFrame(gradients, index=regions, columns=labels)
    eigens = pd.DataFrame({'eigenvalues': lambdas, 'proportion': lambdas / lambdas.sum()}, index=labels)

    if save_threshold is not None:
        retain = eigens[eigens['proportion'] > save_threshold].index.values
        gradients = gradients[retain]

    gradients.to_csv(out_prefix + '_gradient.tsv', sep='\t', float_format=float_fmt)
    eigens.to_csv(out_prefix + '_eigenvalues.tsv', sep='\t', float_format=float_fmt)


def epoch_gradients(cmat_file, reference, out_dir, approach='pca'):
    fname = os.path.split(cmat_file)[1]
    sub_id = fname[:6]

    display(f'{fname}')
    cmat = pd.read_table(cmat_file, index_col=0)
    regions = cmat.columns

    if approach == 'dm':
        gm = _dm_gradients(cmat.values, reference)
    elif approach == 'pca':
        gm = _pca_gradients(cmat.values, reference)

    save_dir = os.path.join(out_dir, sub_id)
    os.makedirs(save_dir, exist_ok=True)
    out_prefix = os.path.join(save_dir, fname.replace('_cmat', '')[:-4])
    save_gradients(gm.aligned_, gm.lambdas_, regions, out_prefix)
    

def dataset_gradient(cmats, out_dir, reference, approach='pca', n_jobs=32):
    os.makedirs(out_dir, exist_ok=True)
    ref_cmat = pd.read_table(reference, index_col=0)
    
    if approach == 'dm':
        gm = _dm_gradients(ref_cmat.values)
    elif approach == 'pca':
        gm = _pca_gradients(ref_cmat.values)
    out_prefix = os.path.join(out_dir, 'reference')
    save_gradients(gm.gradients_, gm.lambdas_, ref_cmat.columns, out_prefix)

    Parallel(n_jobs)(delayed(epoch_gradients)(x, gm.gradients_, out_dir, approach) 
        for x in cmats)

def shuffle_labels(files, out_dir, float_fmt='%1.8f', verbose=False):

    sub_ids = np.unique([re.findall(r'sub-\d+', f)[0] for f in files])
    mappings = []
    for i in sub_ids:

        sub_dir = os.path.join(out_dir, i)
        os.makedirs(sub_dir, exist_ok=True)

        fnames = [f for f in files if i in f]

        indices = list(range(len(fnames)))
        np.random.shuffle(indices)

        for j, k in zip(indices, fnames):
            cmat = pd.read_table(k, index_col=0)
            if verbose:
                print(k, '->', fnames[j])

            out_basename = os.path.basename(fnames[j])
            cmat.to_csv(os.path.join(sub_dir, out_basename), sep='\t', 
                        float_format=float_fmt)
            mappings.append(
                {'original': os.path.basename(k), 'new': out_basename}
            )

    df = pd.DataFrame(mappings)
    df.to_csv(os.path.join(out_dir, 'mappings.csv'), index=False)


def create_reference(dataset_dir, specifier='base', mean='arithmetic'):

    pattern = f'*/*{specifier}*.tsv'
    cmat_files = get_files([dataset_dir, pattern])
    
    cmats = np.array([pd.read_table(i, index_col=0).values for i in cmat_files])
    labels = pd.read_table(cmat_files[0], index_col=0).columns

    if mean == 'arithmetic':
        mean_cmat = np.mean(cmats, axis=0)
    elif mean == 'geometric':
        mean_cmat = mean_riemann(cmats)
    else:
        raise ValueError("mean must be 'arithmetic' or 'geometric'")

    df = pd.DataFrame(mean_cmat, index=labels, columns=labels)
    out = os.path.join(dataset_dir, 'reference_cmat.tsv')
    df.to_csv(out, sep='\t')
    return out

def main():
    config = Config()
    suffix = '-centered'
    cmat_files = get_files([config.connect + suffix, '*/*.tsv'])
    ref_cmat = create_reference(config.connect + suffix, specifier='base', mean='geometric')
    out_dir = os.path.join(config.dataset_dir, f'pca-gradients{suffix}')
    dataset_gradient(cmat_files, out_dir, ref_cmat)

main()
#if __name__ == "__main__":
#    main()
