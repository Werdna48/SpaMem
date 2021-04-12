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

bhv_df['task'] = bhv_df['task'].replace(to_replace=[1, 2, 3], 
value=['s','ns','ts'], )

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
        pred_cols = ['_'.join(i) for i in itertools.product({'ts': ['loc'],
                                                             'ns': ['ori'],
                                                             's': ['ori']}[task],
                                                            ['c', 'u'],
                                                            ['A', 'B', 'C'])]
        crit = bhv_df.loc[idx_sno & idx_task, 'response'].values
        pred = bhv_df.loc[idx_sno & idx_task, pred_cols].values
        allCoefs += [np.array(complexRegression(crit[:,None], pred))]

sNo, task, cued, stim = np.array(list(itertools.product(bhv_df['subjectID'].unique(),
                  bhv_df['task'].unique(),
                  ['c', 'u'],
                  ['A', 'B', 'C']))).T
coef_df = pd.DataFrame(data= list(itertools.product(bhv_df['subjectID'].unique(),
                                                    bhv_df['task'].unique(),
                                                     ['c', 'u'],
                                                     ['A', 'B', 'C'])),
                        columns = ['sNo', 'task', 'cued', 'stim'])

# here we append the length of the regression coefficient to other data (np.abs)
coef_df['coefs'] = np.abs(np.array(allCoefs).reshape(-1, 1))
gav_coef = coef_df.groupby(['task', 'cued', 'stim']).mean().reset_index()

pred = bhv_df['task'].values
crit = bhv_df['response'].values

res = complexRegression(crit, pred)
print(res)
# %%
