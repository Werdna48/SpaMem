'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2021-01-14
-----
Last Modified: 2021-01-29
Modified By: Dragan Rangelov (d.rangelov@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0
Copyright 2019-2021 Dragan Rangelov, The University of Queensland
'''
#===============================================================================
# %% import libraries
#===============================================================================
import numpy as np
import logging
from pathlib import Path
from numpy.core.shape_base import block
import pandas as pd
import itertools
# %% import libraries for plotting
import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()
#===============================================================================
# %% format logger
#===============================================================================
log_format = '%(asctime)s\t%(filename)s\tMSG\t%(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(
    format=log_format,
    datefmt=date_format
)
#===============================================================================
# %% get all data
#===============================================================================
#ROOTPATH = Path('/scratch/im34/DB-SpaMem-01')
#bhvFiles = sorted(ROOTPATH.glob('**/sub-*_task-WM_bhv.tsv.gz'))
# %% load and concatenate all data
#all_df = []
#for fpath in bhvFiles:
#    subNo = int(fpath.stem.split('_')[0].split('-')[-1])
#    logging.warning('Loading subject number {}'.format(subNo))
#    # NOTE: missing responses are indexed as 9999
#    tmp_df = pd.read_csv(fpath, sep = '\t', na_values=9999)
#   tmp_df['subNo'] = subNo
#    all_df += [tmp_df]

# all_df = pd.read_csv('D://Personal//Data//03_Derivatives//allbeh.csv')
all_df = pd.read_csv('C://SpaMem//03_Derivatives//allbeh.csv')
# %% recode task and cue: 
#   1 = spatial [s] / ori, 
#   2 = non spatial [ns] / avg, 
#   3 = true spatial [ts] / loc
all_df['task'] = all_df['cond']
all_df['task'].replace(
    [1, 2, 3],
    ['ori', 'avg', 'loc'],
    inplace=True
) 
#   1 = left, 
#   2 = right
all_df['side'] = all_df['cue']
all_df['side'].replace(
    [1, 2],
    ['left', 'right'],
    inplace=True
) 
# %% Recode X and Y into Cartesian
# X and Y measured from [0,0] top left, screen resolution is 1920x1080
# Therefore centre of screen is [960,540]
#y*-1 because negative y values in this case are actually above the center
#i.e. (x,200) - (960,540) = (x, -340), but 200px is above centre
#     could do (960,540) - (x,200) but then same problem with x
#     decided that y*-1 is easier for consistency otherwise would need
#     x - 960 and 540 - y, which could be more confusing
for point in range(1, 7):
    all_df[f'X_{point}'] =  all_df[f'X_{point}'].values - 960
    all_df[f'Y_{point}'] =  -1*( all_df[f'Y_{point}'].values - 540)  

# %% recode angles 
for col in range(1, 7):
    all_df[f'P_{col}_rad'] = np.arctan2(
        *all_df.loc[:, [f'Y_{col}', f'X_{col}']].values.T
    )
    # we divide the angle by 90 cause these are orientations
    # we add 90 to the angle cause the counting starts from 12 o'clock
    all_df[f'A_{col}_rad'] = (all_df[f'A_{col}'].values + 90) * np.pi / 90
    # normalize the range of orientations to -pi - pi
    all_df[f'A_{col}_rad'] = np.angle(np.exp(all_df[f'A_{col}_rad'] * 1j))
# %% recode angles
# NOTE: for location task, participants were instructed to report only one side 
# so it may not be justified to consider location as a 360 degrees task, 
# but rather a 180 degrees task.
[
    all_df['tarRad'],
    all_df['rspRad']
] =  (all_df[['targetAngle', 'responseAngle']] * np.pi / 90).values.T
# for location task we need to divide by 180 to get to pirad, 
# since we have divided by 90 already, we just need to divide by 2
all_df.loc[all_df['task'] == 'loc', ['tarRad', 'rspRad']] /= 2
# wrap angles
all_df[['tarRad','rspRad']] = np.angle(np.exp(all_df[['tarRad','rspRad']] * 1j))
# compute error magnitude
all_df['errRad'] = np.angle(
    np.exp(all_df['tarRad'] * 1j) 
    / np.exp(all_df['rspRad'] * 1j)
)
# Here's the code I used to check the loc cond
# all_df.loc[(all_df['task'] == 'loc'), ['tarRad', 'P_1_rad' ,'P_2_rad']].head()


# %% rearange columns depending on the cued side 
tar_df = pd.concat([
    all_df.loc[
        (all_df['side'] == side)
        & (all_df['task'] == task),
        [
            col 
            for col in all_df.columns
            if ('rad' in col) 
            and  (
                (('A' in col) and (task != 'loc'))
                or (('P' in col) and (task == 'loc'))
            )
            and (
                ((side == 'left') and (int(col.split('_')[1]) % 2 != 0))
                or ((side == 'right') and (int(col.split('_')[1]) % 2 == 0)) 
            )
        ]
    ].rename(
        columns=dict(
            zip(
                [
                    col 
                    for col in all_df.columns
                    if ('rad' in col) 
                    and  (
                        (('A' in col) and (task != 'loc'))
                        or (('P' in col) and (task == 'loc'))
                    )
                    and (
                        ((side == 'left') and (int(col.split('_')[1]) % 2 != 0))
                        or ((side == 'right') and (int(col.split('_')[1]) % 2 == 0)) 
                    )
                ],
                [
                    '_'.join(col)
                    for col in itertools.product(['TAR'], ['1', '2', '3'])
                ]
            )
        )
    )
    for side in all_df['side'].unique()
    for task in all_df['task'].unique()
])
dis_df = pd.concat([
    all_df.loc[
        (all_df['side'] == side)
        & (all_df['task'] == task),
        [
            col 
            for col in all_df.columns
            if ('rad' in col) 
            and  (
                (('A' in col) and (task != 'loc'))
                or (('P' in col) and (task == 'loc'))
            )
            and (
                ((side == 'left') and (int(col.split('_')[1]) % 2 == 0))
                or ((side == 'right') and (int(col.split('_')[1]) % 2 != 0)) 
            )
        ]
    ].rename(
        columns=dict(
            zip(
                [
                    col 
                    for col in all_df.columns
                    if ('rad' in col) 
                    and  (
                        (('A' in col) and (task != 'loc'))
                        or (('P' in col) and (task == 'loc'))
                    )
                    and (
                        ((side == 'left') and (int(col.split('_')[1]) % 2 == 0))
                        or ((side == 'right') and (int(col.split('_')[1]) % 2 != 0)) 
                    )
                ],
                [
                    '_'.join(col)
                    for col in itertools.product(['DIS'], ['1', '2', '3'])
                ]
            )
        )
    )
    for side in all_df['side'].unique()
    for task in all_df['task'].unique()
])
tmp_df= pd.concat([
    tar_df,
    dis_df
], axis = 1).sort_index()
all_df.drop(
    columns=[
    col 
    for col in all_df.columns
    if '_' in col
], inplace=True)
all_df = all_df.merge(
    tmp_df,
    left_index=True, right_index=True
)
# %% remove missing data
all_df.dropna(inplace = True)   
# remove bad trials
all_df = all_df.loc[all_df['trialOK'] == 1]
#===============================================================================
# %% analyse error magnitudes
#===============================================================================
# compute bins for histograms
NBINS = 10
binIntervals= np.linspace(-np.pi, np.pi, NBINS)
binLabels = np.linspace(-np.pi, np.pi, NBINS + 1)[1:-1]
all_df['errBin'] = pd.cut(all_df['errRad'], binIntervals, labels=binLabels)
# compute frequencies per bin
binFreqs = all_df.groupby([
    'subNo', 
    'task', 
    'cue', 
    'errBin'
]).size().reset_index().rename(columns={0:'binFreq'})
binFreqs['binProp'] = binFreqs.groupby([
    'subNo',
    'task',
    'cue'        
]).apply(lambda x: x['binFreq']/ x['binFreq'].sum()).reset_index()['binFreq']
gavBins = binFreqs.groupby([
    'task',
    'cue',
    'errBin'
]).mean().reset_index()
# %% plot empirical distributions of error magnitudes
fig = plt.figure()
for idx_task, task in enumerate(['ori', 'loc', 'avg']):
    ax = plt.subplot(1, 3, idx_task + 1)
    for cue in [1, 2]:
        ax.bar(
            binLabels, 
            gavBins.loc[
                (gavBins['task'] == task)
                & (gavBins['cue'] == cue),
                'binProp'
            ],
            alpha = .3,
            width = .6
        )
    ax.set_title(task)
    ax.set_ylim(0, .6)
# %%
# TODO: 
# 2. characterise empirical distribution of error magnitudes (SD and M) 
# 3. Mixture distribution model fitting: 
#   a.  fixed mu = 0; same K 
#   b.  variable swap coefficients for cued and uncued side 
#   c.  for averaging task allow variable K for target and swaps, 
#       perhaps penalize if Ktarget > K swaps 
#   d. compute decision weights

#Search function :)
# all_df.loc[(all_df['task'] == 'avg') & (all_df['side'] == 'left'), ['tarRad', 'targetAngle', 'A_1', 'A_2', 'A_3', 'A_1_rad', 'A_3_rad', 'A_5_rad']].head(10)
# use .min()/.max() to find the min and max values for the columns you index

#TODO:
# done 1. Location should be fixed. all_df.loc[(all_df['task'] == 'loc'), ['tarRad', 'P_1_rad', 'P_2_rad']].head(10)
# 2. Ori: decide if we want theta's or phi's and if want (-180 - 180)/(0 - 180)/ or(0 - 360)
# 3. Avg: decide if we want theta's or phi's and if want (-180 - 180)/(0 - 180)/or(0 - 360)
# 4. deltaAngle: Assess if we need to recalculate deltaAngle

#NOTE: How Angles are measured
# targetAngle (previously targetAngle_Polar) is measured 3 o'clock CCW
# A_1:6 (From rundata_A) is measured from 12 o'clock clockwise 

#NOTE:An alternative to the below is to make either the left or right cue'd 
#   angles negative  and do same calcs without the +- 90's/360's (keeping it 0-180)
#   might be preferable to keep left angles similar in magnitude to right angles
#   for the purposes of delta angle calculate we can just use absolute values

#NOTE: Should work to transform ori A_'s(phi's) to 0-360 deg targetAngle(theta)
# neither ori or avg code below has been tested on our data, need to remove the +90 in line 88
#df.loc[(df.side == 1) & (df.cond == 1), 'theta'] = 90 - (-1*df['phi_1']) 
#df.loc[(df.side == 2) & (df.cond == 1) & (df.phi <= 90), 'theta'] = 90 - df['phi_2'] 
#df.loc[(df.side == 2) & (df.cond == 1) & (df.phi > 90), 'theta'] = 360 + (90 - df['phi_2'])
# UNSURE HOW THIS SCALES! UNSURE HOW THIS SCALES! UNSURE HOW THIS SCALES!

#NOTE: Draft for avg 
# Multiply by neg 1 after mean is calcuated, to avoid conflicts with 
# adding negative numbers that could cancel eachother out 
#df.loc[(df.side == 1) & (df.cond == 2), 'theta'] = 90 - (-1*(df['phi_1'] + df['phi_3'] + df['phi_5']).mean(axis=1))
#df.loc[(df.side == 2) & (df.cond == 2) & (df.phi <= 90), 'theta'] = 90 - (df['phi_2'] + df['phi_4'] + df['phi_6']).mean(axis=1)
#df.loc[(df.side == 2) & (df.cond == 2) & (df.phi > 90), 'theta'] = 360 + ((90 - df['phi']) + (90 - df['phi']) + (90 - df['phi'])).mean(axis=1)
# UNSURE HOW THIS SCALES! UNSURE HOW THIS SCALES! UNSURE HOW THIS SCALES!

#NOTE: Important lines to show Dave
# all_df.loc[(all_df['task'] == 'ori') & (all_df['side'] == 'left'), ['tarRad',  'A_1_rad', 'targetAngle', 'A_1']].head(10)
# all_df.loc[(all_df['task'] == 'ori') & (all_df['subID'] == 71), ['targetAngle', 'A_1', 'A_2']].head(10)

#NOTE:Phi to theta calculations (for 0-360 deg)
# Phi angles are measured from 12 CW, theta is measured from 3CCW, however,
# phi has been modded by 180, so right side angles stay the same, but left
# side phi's should be phi+180, if they werent modded. But if we want to keep the mod
# then phi+180 is the same as -phi. Using -phi left cued trials now start from 12 CCW
# so any left theta = 90-(-phi), since -phi is measured CCW like theta, no adjustments need to be made 
# for angles >90
# for any right theta <=90 (which would be 1Q), 1Q = 90, therefore, 90 = phi + theta
# theta = 90 - phi. This wont work for right thetas >90 as Phi is measured CW and theta CCW
# right thetas >90 happen to be 4Q and using the previous formula will give -ve thetas, 
# any negative that can be expressed as a positive theta using theta_pos = 360 + theta_neg
# we can then use our negative theta calcuated above to find a positive theta that
# will be between (270-360deg)


#NOTE:theta = phi (-180-180) deg or theta = phi (0-180) deg calcualtions
# calculating our theta's as phi's makes the most sense if we care about 
# left and right sided error magnitudes to be similar (if we wanted this but in terms of 3 o'clock
# I would reccomended mod 180 the values given in the 0-360 degree calcs; but aslong as both are in terms of eachother it shouldnt matter)
# if anything it might be easier to keep them as phi's to distinguish between left and right cues more intuitively
# Basically theta = phi (-180-180) would just be making left sided or right sided cues -ve
# I would say that left = negative makes most sense as phi is measured CW, so Q1 and Q4
# are measured first and they are on the right, which rightfully means that the left should be negative
# applying polar coordinate logic here. Otherwise we dont do that and set theta to phi and call it a day
# no transforms are neccessary other than multiplying left sided phis by -1 if we
# decided for (-180-180)