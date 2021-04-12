# %% Libraries
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from scipy import stats
from scipy.stats import circstd
from statsmodels.stats.anova import AnovaRM
import glob
import os
import pingouin as pg
# %%
#Data Collation start - Code used to create the dfs and collate via Massive
paths = sorted(glob.glob(
    'D://Personal//Data//02_Rawdata//**//beh//**.tsv', recursive=True
))

paths = sorted(glob.glob(
    'C://PatDat//02_RawData//**//beh//**.tsv', recursive=True
))

all_df = []
for path in paths:
    tmp_df = pd.read_csv(
        path,
        sep = '\t',
        na_values=9999
    )
    tmp_df['subID'] = int(Path(path).stem.split('_')[0].split('-')[-1])
    all_df += [tmp_df]
all_df = pd.concat(all_df) 

trialok = all_df['trialOK'] == 1
ok_all_df = all_df[trialok]
ok_all_df.dropna()
#End of Data Collation - The following code can be used in conjunction with the next segment but I
#made a collated file for ease of use on my end

#Start of Basic Behavioural analysis - Set up DFS and ANOVA, in conjunction with a plot
allbeh = pd.read_csv('D:/Personal/Data/03_Derivatives/allbeh.csv')

avg_df = ok_all_df.groupby([
    'subID',
    'cond',
    'cue'
])['deltaAngle'].apply(circstd).reset_index().rename(columns = {
    'deltaAngle':'cstd'
})
gav_df = avg_df.groupby(['cond', 'cue'])['cstd'].mean().reset_index()
cnd_df = avg_df.groupby(['cond'])['cstd'].mean().reset_index()

avg_df.to_csv('D:/Personal/Data/03_Derivatives/avgbeh.csv')

model = print(pg.rm_anova(dv='cstd', within=['cond', 'cue'],
        subject='subID', data=avg_df, detailed=True))

print(AnovaRM(data=avg_df, depvar='cstd', 
    subject='subID', within=['cond', 'cue']).fit())

# TODO: replace yerr with actual error value later

plot_df = avg_df.groupby(['cue', 'cond'])['cstd'].mean().unstack(0)

plot_df.T
#Important to not Cond 1 = Spatial, 2 = NonSpatial, 3 = True Spatial
plot_df.index =['Spatial', 'NonSpatial', 'TrueSpatial']
# Cue 1 = L, 2 = R
plot_df.columns =['Left', 'Right']

cerr = 0.094
qerr = 0.312

plot_df.plot.bar().set_ylabel('Circular SD').errorbar(
    yerr=['cerr', 'qerr'])
#End of Basic Behavioural Analysis 
# %%
