# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:18:43 2024

@author: jstef
"""

from helpers import load_and_format_IDP_or_SAW_data, load_and_format_SAT_data, load_and_format_SWM_data, load_and_format_Raven_data
from os import getcwd, path, chdir, listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# set paths
homedir = path.expanduser("~")
reppath = path.join(homedir,"Nextcloud","SESYN","_B09","SATPD2_AUD","plandepth_aud")
datadir = path.join(reppath,"data")
chdir(reppath)

datadir_hc  =  path.join(datadir,"HC")  # change to correct paths
datadir_aud =  path.join(datadir,"AUD")   
datadir_tud =  path.join(datadir,"TUD")
datadir_aud_tud =  path.join(datadir,"AUD_and_TUD")

grouplabels = ['HC','AUD','TUD','AUD_and_TUD']

# get rid of deprecation warning
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)



#%% load SAT behavioral data
    
# SAT data
filenames = ["space_adventure_pd-results.json",
             "space_adventure_pd_inv-results.json"]    # posible filenames of SAT logfiles

stimuli_hc,      mask_hc,      responses_hc,      rts_hc,      scores_hc,      conditions_hc,      ids_hc,      bal_hc,      noTrainings_hc      = load_and_format_SAT_data(datadir_hc,      filenames, discard_training_blocks=False)
stimuli_aud,     mask_aud,     responses_aud,     rts_aud,     scores_aud,     conditions_aud,     ids_aud,     bal_aud,     noTrainings_aud     = load_and_format_SAT_data(datadir_aud,     filenames, discard_training_blocks=False)
stimuli_tud,     mask_tud,     responses_tud,     rts_tud,     scores_tud,     conditions_tud,     ids_tud,     bal_tud,     noTrainings_tud     = load_and_format_SAT_data(datadir_tud,     filenames, discard_training_blocks=False)
stimuli_aud_tud, mask_aud_tud, responses_aud_tud, rts_aud_tud, scores_aud_tud, conditions_aud_tud, ids_aud_tud, bal_aud_tud, noTrainings_aud_tud = load_and_format_SAT_data(datadir_aud_tud, filenames, discard_training_blocks=False)

df_sat = pd.DataFrame(list(zip(ids_hc + ids_aud + ids_tud + ids_aud_tud, 
                               torch.cat((scores_hc[:,-1,-1], 
                                          scores_aud[:,-1,-1], 
                                          scores_tud[:,-1,-1], 
                                          scores_aud_tud[:,-1,-1])).tolist(),
                               (torch.cat((rts_hc[:,:,0].mean(dim=1), 
                                          rts_aud[:,:,0].mean(dim=1), 
                                          rts_tud[:,:,0].mean(dim=1), 
                                          rts_aud_tud[:,:,0].mean(dim=1))) / 1000).tolist())), 
                      columns=['ID','SAT_score', 'SAT_RT'])


#%% load IDP and SAW data

df_list_idp = []
df_list_saw = []
filenames_idp = ['identical_pictures_task-results.json']
filenames_saw = ['spot_a_word_task-results.json']
for i,gd in enumerate(grouplabels):
    groupdir_path = path.join(datadir,gd)
    tmp_idp = load_and_format_IDP_or_SAW_data(groupdir_path,filenames_idp,'IDP')
    tmp_saw = load_and_format_IDP_or_SAW_data(groupdir_path,filenames_saw,'SAW')
    tmp_idp['group'] = gd
    tmp_saw['group'] = gd
    df_list_idp.append(tmp_idp)
    df_list_saw.append(tmp_saw)
df_idp = pd.concat(df_list_idp, ignore_index=True)
df_saw = pd.concat(df_list_saw, ignore_index=True)


#%% load SWM data

df_list = []
filenames = ['spatial_working_memory_task-results.json','spatial_working_memory_task_inv-results.json']

for i,gd in enumerate(grouplabels):
    groupdir_path = path.join(datadir,gd)
    tmp = load_and_format_SWM_data(groupdir_path,filenames)
    tmp['group'] = gd
    df_list.append(tmp)
df_swm = pd.concat(df_list, ignore_index=True)


#%% load Raven data

df_list = []
filenames = ['raven_task-results.json']

for i,gd in enumerate(grouplabels):
    groupdir_path = path.join(datadir,gd)
    tmp = load_and_format_Raven_data(groupdir_path,filenames)
    tmp['group'] = gd
    df_list.append(tmp)
df_raven = pd.concat(df_list, ignore_index=True)


#%% Merge data

df = pd.merge(df_sat, df_raven, on="ID")
df = pd.merge(df, df_idp, on=["ID","group"])
df = pd.merge(df, df_saw, on=["ID","group"])
df = pd.merge(df, df_swm, on=["ID","group"])

#%% EXCLUSION

excl_ids = []

# SAT Score <= 650 (mean outcome of random behavior)
threshold = 650
excl_ids_hc      = [ ids_hc[e]      for e in (scores_hc[:,-1,-1]      <= threshold).nonzero() ]
excl_ids_aud     = [ ids_aud[e]     for e in (scores_aud[:,-1,-1]     <= threshold).nonzero() ]
excl_ids_tud     = [ ids_tud[e]     for e in (scores_tud[:,-1,-1]     <= threshold).nonzero() ]
excl_ids_aud_tud = [ ids_aud_tud[e] for e in (scores_aud_tud[:,-1,-1] <= threshold).nonzero() ]
excl_ids += excl_ids_hc + excl_ids_aud + excl_ids_tud + excl_ids_aud_tud

# Raven: CORR + CORR_Intr <= 1 (mean outcome of random behavior: 1.75)
threshold = 1
excl_ids += df_raven[df_raven.RAV_CORR+df_raven.RAV_CORR_Instr <= threshold].ID.to_list()
# Raven: CORR <= 1 (mean outcome random behavior: 1.5)
#threshold = 1
#excl_ids += df_raven[df_raven.RAV_CORR <= threshold].ID.to_list()

# IDP: CORR <= 9 (mean outcome of random behavior: 9.2)
threshold = 9
excl_ids += df_idp[df_idp.IDP_CORR <= threshold].ID.to_list()

# SWM: SWM_CORR <= 48 (mean outcome of random behavior)
threshold = 48
excl_ids += df_swm[df_swm.SWM_CORR <= threshold].ID.to_list()

# SWM: SER_ACC <= 50 (mean outcome of random behavior)
threshold = 50
excl_ids += df_swm[df_swm.SER_ACC <= threshold].ID.to_list()

# SAW: CORR <= 7 (mean outcome of random behavior)
threshold = 7
excl_ids += df_saw[df_saw.SAW_CORR <= threshold].ID.to_list()

# get unique IDs
excl_ids = list(set(excl_ids))

# store to file
pd.DataFrame({'ID':excl_ids}).to_csv(path.join(datadir,'exclusion_ids.csv'))

# EXCLUDE from merged dataframe
df = df[~df.ID.isin(excl_ids)]

print("HC:     Excluded: "+str(len(ids_hc)     -len(set(ids_hc)     -set(excl_ids)))+" | Included: "+str(len(list(set(ids_hc)     -set(excl_ids)))))
print("AUD     Excluded: "+str(len(ids_aud)    -len(set(ids_aud)    -set(excl_ids)))+" | Included: "+str(len(list(set(ids_aud)    -set(excl_ids)))))
print("TUD     Excluded: "+str(len(ids_tud)    -len(set(ids_tud)    -set(excl_ids)))+" | Included: "+str(len(list(set(ids_tud)    -set(excl_ids)))))
print("AUD+TUD Excluded: "+str(len(ids_aud_tud)-len(set(ids_aud_tud)-set(excl_ids)))+" | Included: "+str(len(list(set(ids_aud_tud)-set(excl_ids)))))

    
#%% explore validity of Raven Data
from scipy.stats import binom 

# load B3 dataset for comparison
df_B3_raven = pd.read_csv(path.join(datadir,"Raven_B3_Sophia.csv"))
df_B3_raven.rename(columns={"Raven_RT": "RAV_RT", "Raven_RESP_CORR": "RAV_CORR", "Raven_Example_trials_num_corr":"RAV_CORR_Instr"}, inplace=True)
df_B3_raven.RAV_RT /= 1000 # transform RTs to seconds

for i,dataset in enumerate([df_raven,df_B3_raven]):
    
    project = ['B9','B3'][i]
    ns = len(dataset) # number of subjects
    n  = 12     # number of trials
    x_values = list(range(n + 1))
    p  = 0.125  # prob of success
    # list of pmf values * subjects = expected nr of subjects
    dist = [ns * binom.pmf(r, n, p) for r in x_values ] 
    
    # plot hist of sum_correct
    plt.bar(x_values, dataset.RAV_CORR.value_counts().sort_index().tolist(),label='dataset')
    
    # add plot of expected dist for random behavior
    plt.bar(x_values, dist, fill=False, label='expectation of random behavior')
    
    ax = plt.gca()
    ax.set_xlabel('Number of correct trials')
    ax.set_ylabel('subjects')
    plt.legend()
    plt.title(project + ' Raven Task - Histogram of Correct Trials')
    plt.show()
    
    # plot hist of RT
    plt.hist(dataset.RAV_RT.tolist())
    plt.title(project + ' Raven Task - Histogram of Mean RT')
    ax = plt.gca()
    ax.set_xlabel('RT[s]')
    ax.set_ylabel('subjects')
    plt.show()
    
    # plot hist of intro trials correct
    plt.bar([0,1,2], dataset.RAV_CORR_Instr.value_counts().sort_index().tolist())
    plt.xlabel('Number of correct trials')
    plt.ylabel('subjects')
    plt.title(project + ' Raven Task - Histogram of Correct Intro Trials')
    plt.show()

#%% load pd inference results from npz
# m_prob_aud = np.load("plandepth_stats_aud.npz", allow_pickle=True)['arr_1']
# m_prob_tud = np.load("plandepth_stats_tud.npz", allow_pickle=True)['arr_1']
# m_prob_aud_tud = np.load("plandepth_stats_aud_tud.npz", allow_pickle=True)['arr_1']
# m_prob_hc  = np.load("plandepth_stats_hc.npz", allow_pickle=True)['arr_1']

#%%
# # load parameter csv file
# pars_df = pd.read_csv('pars_post_samples.csv', index_col=0, dtype={'ID':object})
# pars_IDorder_aud = pars_df[pars_df.group_label=='AUD'].groupby(by=['ID','subject']).size().reset_index().subject.to_numpy()
# pars_IDorder_tud = pars_df[pars_df.group_label=='TUD'].groupby(by=['ID','subject']).size().reset_index().subject.to_numpy()
# pars_IDorder_aud_tud = pars_df[pars_df.group_label=='AUD_and_TUD'].groupby(by=['ID','subject']).size().reset_index().subject.to_numpy()
# pars_IDorder_hc  = pars_df[pars_df.group_label=='HC'].groupby(by=['ID','subject']).size().reset_index().subject.to_numpy()


