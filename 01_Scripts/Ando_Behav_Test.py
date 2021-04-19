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
import itertools
# %%
#Currently working on using the Complex Regression
# bhv_df = pd.read_csv('D:/Personal/Data/03_Derivatives/allbeh.csv')

RAWPATH = Path("C://PatDat//02_RawData")

# Forgot to put the data set on my laptop so use this instead for now
paths = sorted(RAWPATH.glob('**/*.tsv'))

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
bhv_df = all_df[trialok]
bhv_df.dropna()
# Delete this cell and comment after you finish tofday


bhv_df = bhv_df.rename(columns = {
    'subID':'subjectID', 'cond':'task', 'responseAngle':'response'})


# Currently do not have the following columns, need to ask Dragan what these are
# ori_c_A', 'ori_c_B', 'ori_c_C', 'ori_u_A', 'ori_u_B', 'ori_u_C
# I assume that these are location and orientaion information of tagets/distractors
# that are created via the XYA columns in our data set

# @Author: Dragan Rangelov <uqdrange> <- Where the complex Regression is sourced from
# @Date:   03-6-2019
# @Email:  d.rangelov@uq.edu.au
# @Last modified by:   uqdrange
# @Last modified time: 03-6-2019
# @License: CC-BY-4.0
def complexRegression(crit, pred):
    '''
    Compute regression coefficients for predictors
    args
    ====
    crit - dependent variable, N trials x 1, -pi to pi
    pred - independent variables, N trials x M predictors, -pi to pi
    returns
    =======
    vector of coefficients 1 X M
    '''
    pred = np.exp(pred * 1j)
    crit = np.exp(crit * 1j)
    coefs = (np.asmatrix(np.asmatrix(pred).H
                        * np.asmatrix(pred)).I
             * (np.asmatrix(pred).H
                * np.asmatrix(crit)))
    return coefs

allCoefs = []
for sno in bhv_df['subjectID'].unique():
    idx_sno = bhv_df['subjectID'] == sno
    for task in bhv_df['task'].unique():
        idx_task = bhv_df['task'] == task
<<<<<<< HEAD
        for cue in bhv_df['cue'].unique():
            idx_cue = bhv_df['cue'] == cue
            pred_cols = ['_'.join(i) for i in itertools.product({'ts': ['loc'],
                                                                 'ns': ['ori'],
                                                                 's': ['ori']}[task],
                                                                ['c', 'u'],
                                                                ['A', 'B', 'C'])]
            crit = bhv_df.loc[idx_sno & idx_task & idx_cue, 'response'].values
            pred = bhv_df.loc[idx_sno & idx_task & idx_cue, pred_cols].values
            allCoefs += [np.array(complexRegression(crit[:,None], pred))]

=======
        pred_cols = ['_'.join(i) for i in itertools.product({3: ['loc'],
                                                             2: ['ori'],
                                                             1: ['ori']}[task],
                                                            ['c', 'u'],
                                                            ['A', 'B', 'C'])]
        crit = bhv_df.loc[idx_sno & idx_task, 'response'].values
        pred = bhv_df.loc[idx_sno & idx_task, pred_cols].values
        allCoefs += [np.array(complexRegression(crit[:,None], pred))]

sno, task, cued, stim = np.array(list(itertools.product(bhv_df['subjectID'].unique(),
                  bhv_df['task'].unique(),
                  ['c', 'u'],
                  ['A', 'B', 'C']))).T
>>>>>>> 8080451a44ad4fa6325151e3c10fc84e6756936f
coef_df = pd.DataFrame(data= list(itertools.product(bhv_df['subjectID'].unique(),
                                                    bhv_df['task'].unique(),
                                                    bhv_df['cue'].unique(),
                                                    ['c', 'u'],
                                                    ['A', 'B', 'C'])),
                        columns = ['sNo', 'task', 'side', 'cued', 'stim'])
# here we append the length of the regression coefficient to other data (np.abs)
coef_df['abs_Theta'] = np.abs(np.array(allCoefs).reshape(-1, 1))
coef_df['cos_Theta'] = np.cos(np.angle(np.array(allCoefs).reshape(-1, 1)))
coef_df['weighted_Theta'] = coef_df['abs_Theta'] * coef_df['cos_Theta']
gav_coef = coef_df.groupby(['task', 'cued', 'stim', 'side']).mean().reset_index()
# %%

# Should "work" but as the base DF is, it requires too much memory to analyse 
# but it was only the memory dump error and not a code running error so 
# the fucntion should work
crit = bhv_df['response']
pred = bhv_df['cue']

<<<<<<< HEAD
output = complexRegression(crit, pred)
print(output)
=======
pred = bhv_df['task'].values
crit = bhv_df['response'].values

res = complexRegression(crit, pred)
print(res)
>>>>>>> 8080451a44ad4fa6325151e3c10fc84e6756936f
# %%
