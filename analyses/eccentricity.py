import os
import numpy as np
import pandas as pd

from config import Config
from utils import get_files, load_gradients, schaefer1000_roi_ix

FIG_DIR = os.path.join(Config().figures, 'eccentricity')
os.makedirs(FIG_DIR, exist_ok=True)


def _read_gradients(fname, k, sub,  epoch):
    df = load_gradients(fname, k)
    for col, val in zip(['sub', 'epoch'], [sub,  epoch]):
        df[col] = val
    df['roi_ix'] = schaefer1000_roi_ix()
    return df


def load_subject_gradients(k=3):
    config = Config()
    gradient_dir = config.gradients
    full_files = get_files([gradient_dir, f'*/*full*_gradient.tsv'])

    subject_gradients = []
    for g in full_files:
        sub, _, epoch, _ = os.path.basename(g).split('_')
        subject_gradients.append(_read_gradients(g, k, sub, epoch))

    subject_gradients = pd.concat(subject_gradients)
    return subject_gradients


def compute_eccentricity(data):
    def _get_dists(x):
        grads = x.filter(like='g')
        centroid = grads.mean().values
        x['distance'] = np.linalg.norm(grads.values - centroid, axis=1)
        return x

    return data.groupby(['sub', 'epoch']).apply(_get_dists)


def main():

    config = Config()
    
    gradients = load_subject_gradients(config.k)
    gradients = compute_eccentricity(gradients)
    gradients.to_csv(os.path.join(config.results, 'subject_gradients.tsv'), sep='\t', index=False)

main()
