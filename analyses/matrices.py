import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import Config
from utils import get_files
config = Config()

pjoin = os.path.join

cd = pjoin(config.connect + '/')

def all_participant_matrix(connectivity_directory):
    out_dir = pjoin(config.figures, 'matrices')
    base_out_dir = pjoin(config.figures, 'matrices', 'base')
    early_out_dir = pjoin(config.figures, 'matrices', 'early')
    late_out_dir = pjoin(config.figures, 'matrices', 'late')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(base_out_dir, exist_ok=True)
    os.makedirs(early_out_dir, exist_ok=True)
    os.makedirs(late_out_dir, exist_ok=True)
    
    subid = ['sub-01', 'sub-02', 'sub-03', 'sub-05', 'sub-06',  'sub-07',  'sub-10',  'sub-11',  'sub-12',  'sub-13',   'sub-14',   'sub-15',   'sub-16',  'sub-17',  'sub-18',  'sub-19',  'sub-20',  'sub-21',  'sub-22',  'sub-23',  'sub-24',  'sub-25',  'sub-26',  'sub-27',  'sub-28',  'sub-29',  'sub-30',  'sub-31',  'sub-32',  'sub-33',  'sub-34',  'sub-35',  'sub-36',  'sub-37']

    all_subjects_cmat_data = []
    for sub in subid:
        cmat_data = get_files(connectivity_directory + sub + "/" + '*full*.tsv') 
        for i in cmat_data:
            all_subjects_cmat_data.append(i)
    global early_cmats
    global base_cmats
    global late_cmats
    base_cmats = []
    early_cmats = []
    late_cmats = []
    for i in all_subjects_cmat_data:
        if i[-9:] == "_base.tsv":
            df = pd.read_table(i)
            df = df.drop([0], axis=0)
            df = df.drop(['Unnamed: 0'], axis=1)
            array = df.to_numpy()
            base_cmats.append(array)
            mat, labels = df, df.columns.values.tolist()
            ax = sns.heatmap(mat, square=True, vmin=-.8, vmax=.8, center=0, cmap=sns.diverging_palette(220, 20, n=200), cbar=False)
            plt.savefig(pjoin("/ssd1920/lamplab/projects/stat/statisticalmanifolds/figures/fig-components-pngs-k3/matrices/base/", i[113:119]))
           #plt.savefig(f"/ssd1920/lamplab/projects/stat/statisticalmanifolds/figures/fig-components-pngs-k3/matrices/base/{i[113:120]}_connectivity.png")
        elif i[-9:] == "early.tsv":
            df = pd.read_table(i)
            df = df.drop([0], axis=0)
            df = df.drop(['Unnamed: 0'], axis=1)
            array = df.to_numpy()
            early_cmats.append(array)
            mat, labels = df, df.columns.values.tolist()
            ax = sns.heatmap(mat, square=True, vmin=-.8, vmax=.8, center=0, cmap=sns.diverging_palette(220, 20, n=200), cbar=False)
            #plt.savefig(f"/ssd1920/lamplab/projects/stat/statisticalmanifolds/figures/fig-components-pngs-k3/matrices/early/{i[113:120]}_connectivity.png")
            plt.savefig(pjoin("/ssd1920/lamplab/projects/stat/statisticalmanifolds/figures/fig-components-pngs-k3/matrices/early/", i[113:119]))
        elif i[-9:] == "_late.tsv":
            df = pd.read_table(i)
            df = df.drop([0], axis=0)
            df = df.drop(['Unnamed: 0'], axis=1)
            array = df.to_numpy()
            late_cmats.append(array)
            mat, labels = df, df.columns.values.tolist()
            ax = sns.heatmap(mat, square=True, vmin=-.8, vmax=.8, center=0, cmap=sns.diverging_palette(220, 20, n=200), cbar=False)
            #plt.savefig(f"/ssd1920/lamplab/projects/stat/statisticalmanifolds/figures/fig-components-pngs-k3/matrices/late/{i[113:120]}_connectivity.png")
            plt.savefig(pjoin("/ssd1920/lamplab/projects/stat/statisticalmanifolds/figures/fig-components-pngs-k3/matrices/late/", i[113:119]))
#all_participant_matrix(cd)

mean_base_connectivity = np.mean([base_cmats[0], base_cmats[1], base_cmats[3], base_cmats[4], base_cmats[5], base_cmats[6], base_cmats[7], base_cmats[8], base_cmats[9], base_cmats[10], base_cmats[11], base_cmats[12], base_cmats[13], base_cmats[14], base_cmats[15], base_cmats[16], base_cmats[17], base_cmats[18], base_cmats[19], base_cmats[20], base_cmats[21], base_cmats[22], base_cmats[23], base_cmats[24], base_cmats[25], base_cmats[26], base_cmats[27], base_cmats[28], base_cmats[29], base_cmats[30], base_cmats[31], base_cmats[32], base_cmats[33]], axis=0)   
early_base_connectivity = np.mean([early_cmats[0], early_cmats[1], early_cmats[3], early_cmats[4], early_cmats[5], early_cmats[6], early_cmats[7], early_cmats[8], early_cmats[9], early_cmats[10], early_cmats[11], early_cmats[12], early_cmats[13], early_cmats[14], early_cmats[15], early_cmats[16], early_cmats[17], early_cmats[18], early_cmats[19], early_cmats[20], early_cmats[21], early_cmats[22], early_cmats[23], early_cmats[24], early_cmats[25], early_cmats[26], early_cmats[27], early_cmats[28], early_cmats[29], early_cmats[30], early_cmats[31], early_cmats[32], early_cmats[33]], axis=0)   
late_base_connectivity = np.mean([late_cmats[0], late_cmats[1], late_cmats[3], late_cmats[4], late_cmats[5], late_cmats[6], late_cmats[7], late_cmats[8], late_cmats[9], late_cmats[10], late_cmats[11], late_cmats[12], late_cmats[13], late_cmats[14], late_cmats[15], late_cmats[16], late_cmats[17], late_cmats[18], late_cmats[19], late_cmats[20], late_cmats[21], late_cmats[22], late_cmats[23], late_cmats[24], late_cmats[25], late_cmats[26], late_cmats[27], late_cmats[28], late_cmats[29], late_cmats[30], late_cmats[31], late_cmats[32], late_cmats[33]], axis=0)   

ax = sns.heatmap(mean_base_connectivity, square=True, vmin=-.8, vmax=.8, center=0, cmap=sns.diverging_palette(220, 20, n=200))
plt.savefig(f"/ssd1920/lamplab/projects/stat/statisticalmanifolds/figures/fig-components-pngs-k3/matrices/base/mean_connectivity.png")

ax = sns.heatmap(early_base_connectivity, square=True, vmin=-.8, vmax=.8, center=0, cmap=sns.diverging_palette(220, 20, n=200))
plt.savefig(f"/ssd1920/lamplab/projects/stat/statisticalmanifolds/figures/fig-components-pngs-k3/matrices/early/mean_connectivity.png")

ax = sns.heatmap(late_base_connectivity, square=True, vmin=-.8, vmax=.8, center=0, cmap=sns.diverging_palette(220, 20, n=200))
plt.savefig(f"/ssd1920/lamplab/projects/stat/statisticalmanifolds/figures/fig-components-pngs-k3/matrices/early/mean_connectivity.png")

