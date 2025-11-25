import numpy as np
import torch
from torch import zeros, ones
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from os.path import expanduser, isdir, join
from os import listdir, walk, path
import json
from scipy import io
import re
from statistics import stdev, mean
import math
from pandas.api.types import is_numeric_dtype

def mean_str(col):
    # aggregate function for pandas dataframe that computes mean values for numeric dtypes 
    # and for other dtypes only aggregates if there is only one unique value per group
    if is_numeric_dtype(col):
        return col.mean()
    else:
        return col.unique()[0] if col.nunique() == 1 else np.NaN


# function for plotting asymetric errorbars
def errorplot(*args, **kwargs):
    """plot asymetric errorbars"""
    subjects = args[0]
    values = args[1].values
    
    unique_subjects = np.unique(subjects)
    nsub = len(unique_subjects)
    
    values = values.reshape(-1, nsub)
    
    quantiles = np.percentile(values, [5, 50, 95], axis=0)
    
    low_perc = quantiles[0]
    up_perc = quantiles[-1]
    
    x = unique_subjects
    y = quantiles[1]

    assert np.all(low_perc <= y)
    assert np.all(y <= up_perc)
    
    kwargs['yerr'] = [y-low_perc, up_perc-y]
    kwargs['linestyle'] = ''
    kwargs['marker'] = 'o'
    
    plt.errorbar(x, y, **kwargs)
    
# function for mapping strings ('high', 'low') to numbers 0, 1
def map_noise_to_values(strings):
    """mapping strings ('high', 'low') to numbers 0, 1"""
    for s in strings:
        if s[0] == 'high':
            yield 1
        elif s[0] == 'low':
            yield 0
        else:
            yield np.nan

    
def load_and_format_SAT_data(datadir, filenames, discard_training_blocks = True, exclusionnames=["_incomplete_datasets"], exclusion_ids=[], inclusion_ids=[]):
    
    # search datadir and subdirectories (without dirnames in exclusionnames) for files named like filename
    fnames = listdir(datadir)
    fnames = [f for f in fnames if f not in exclusionnames]
    fnames = [f for f in fnames if re.search(r"\d{5}",f) != None] # takes dirs named after ID (5 digits)
    
    # handle exclusion ids (exclude IDs that are in exclusion_ids)
    exclusion_ids = [a for a in fnames if a in exclusion_ids]
    fnames = [a for a in fnames if a not in exclusion_ids]

    # handle inclusion ids (only use IDs in inclusion_ids). If ID is both in exclusion and inclusion list, it will be excluded.
    inclusion_ids = [a for a in fnames if a in inclusion_ids]
    if inclusion_ids != []:
        fnames = [a for a in fnames if a in inclusion_ids]
    
    runs = len(fnames)  # number of subjects

    if discard_training_blocks:   # number of mini blocks in each run
        start_block = 20
        mini_blocks = 120
    else:
        start_block = 0
        mini_blocks = 140
    max_trials = 3  # maximal number of trials within a mini block
    max_depth = 3  # maximal planning depth

    na = 2  # number of actions
    ns = 6 # number of states/locations
    no = 5 # number of outcomes/rewards

    responses = zeros(runs, mini_blocks, max_trials)
    states = zeros(runs, mini_blocks, max_trials+1, dtype=torch.long)
    scores = zeros(runs, mini_blocks, max_depth)
    conditions = zeros(2, runs, mini_blocks, dtype=torch.long)
    confs = zeros(runs, mini_blocks, 6, dtype=torch.long)
    balance_cond = zeros(runs)
    rts = zeros(runs, mini_blocks, 3, dtype=torch.float64)
    ids = []
    noPatternTrainings = []
    
    fnames.sort() # sort IDs such that order during loop is always the same (especially between different OS)
    
    for i,ID in enumerate(fnames):
        if ID not in exclusion_ids and (inclusion_ids == [] or ID in inclusion_ids):
            iddir = join(datadir,ID)
            logfiles = listdir(iddir)
            logfiles = [f for f in logfiles if f in filenames]
            if len(logfiles)>1: print("WARNING: Multiple logfiles for ID "+ID+". First file will be used.")
            logfile = logfiles[0]
            f = join(iddir, logfile)
                              
            # check for exclusion (gameover by negative points or incomplete data)
            
            with open(f,"r", encoding='utf-8-sig') as read_file:
                # assume json file
                tmp = json.loads(json.load(read_file)['data'])
            if all(flag <= 0 for (_, _, flag) in tmp['points']) or len(tmp['points']) != len(tmp['conditions']['noise']): print("WARNING: exclusion criteria met for ID "+ID+"! Will be skipped.")
            else:     
                responses[i] = torch.from_numpy(np.array(tmp['responses']['actions']) -1)[start_block:,]
                states[i] = torch.from_numpy(np.array(tmp['states']) - 1).long()[start_block:,]
                confs[i] = torch.from_numpy(np.array(tmp['planetConfigurations']) - 1).long()[start_block:,]
                starting_points = tmp['startingPoints']
                if discard_training_blocks:
                    # compute gain during training blocks to substract from cumulative scores
                    training_gain = torch.from_numpy(np.array(tmp['points']))[start_block-1,-1]-starting_points
                else:
                    training_gain = 0
                scores[i] = torch.from_numpy(np.array(tmp['points']))[start_block:,] - training_gain
                strings = tmp['conditions']['noise']
                
                conditions[0, i] = torch.tensor(np.unique(strings, return_inverse=True)[1]*(-1) + 1 , dtype=torch.long)[start_block:]  # "low" -> 0 | "high" -> 1
                conditions[1, i] = torch.from_numpy(np.array(tmp['conditions']['notrials'])).long()[start_block:] 
                
                rts[i] = torch.from_numpy(np.array(tmp['responses']['RT']))[start_block:,] 
                balance_cond[i] = (tmp['balancingCondition'] - 1) 
                
                ids.append(ID)
                noPatternTrainings.append(tmp['noPatternTrainings'])
    
    states[states < 0] = -1
    confs = torch.eye(no)[confs]

    # define dictionary containing information which participants recieved on each trial
    stimuli = {'conditions': conditions,
               'states': states, 
               'configs': confs}

    mask = ~torch.isnan(responses)
    
    return stimuli, mask, responses, rts, scores, conditions, ids, balance_cond, noPatternTrainings



def load_and_format_Raven_data(datadir,filenames,exclusion_ids=[]): 

    max_trials = 12
    
    df_list = []
    
    fnames = listdir(datadir)
    
    fnames.sort() # sort IDs such that order during loop is always the same (especially between different OS)
    
    for i,ID in enumerate(fnames):
        if ID not in exclusion_ids:
            rts = []
            rt_task = 0 # sum of all task rts inkl tooFast trials
            n_corr_ex = 0 # no of correct examples
            n_corr    = 0 # no of correct responses
            n_total   = 0 # no of responses
            n_missing = 0
            n_tooFast = 0
            
            iddir = join(datadir,ID)
            logfiles = listdir(iddir)
            logfiles = [f for f in logfiles if f in filenames]
            if len(logfiles)>1: print("WARNING: Multiple logfiles for ID "+ID+". First file will be used.")
            logfile = logfiles[0]
            f = join(iddir, logfile)
            with open(f,"r", encoding='utf-8-sig') as read_file:
                # assume json file
                tmp = json.loads(json.load(read_file)['data'])
            for el in tmp:
                if el['sender']=="instructions":
                    rt_instruction = el['duration']
                elif el['sender'] in ['example1','example2']:
                    if el['correct']==True: n_corr_ex += 1
                elif "test" in el['sender']:
                    if el['ended_on']=='response':
                        # exclude RTs < 150ms and missing values
                        if el['duration'] is None: 
                            n_missing += 1
                            # print('Missing data for '+ID+' in trial '+el['sender_id']+' [duration]')
                        elif el['duration'] <= 150: n_tooFast += 1; rt_task += el['duration']
                        else:
                            n_total += 1
                            rt_task += el['duration']
                            if el['correct']==True: n_corr += 1
                            rts.append(el['duration'])
                    elif el['ended_on']=='timeout': n_total += 1 # count timeout as error
                    
            # create row for subject-wise dataframe
            entry = pd.DataFrame()
            entry['ID']         = [ID ]
            entry['RAV_MaxCORR']    = [max_trials ]
            entry['RAV_CORR']       = [n_corr ]
            entry['RAV_ERR']        = [n_total - n_corr ]
            entry['RAV_Missing']    = [n_missing ]
            entry['RAV_TooFast']    = [n_tooFast ]
            entry['RAV_PER']        = [n_corr / max_trials * 100 ]
            entry['RAV_ACC']        = [n_corr / n_total    * 100 ]
            entry['RAV_RT']         = [mean(rts)  / 1000 ]
            entry['RAV_RT_SD']      = [stdev(rts) / 1000 ]
            entry['RAV_RT_Instr']   = [rt_instruction / 1000 ]
            entry['RAV_RT_Task']    = [rt_task / 1000 ]
            entry['RAV_CORR_Instr'] = [n_corr_ex ]
            
            df_list.append(entry)
    
    return pd.concat(df_list, ignore_index=True)




def load_and_format_IDP_or_SAW_data(datadir,filenames,taskname, exclusion_ids=[]): 
    t = taskname
    max_trials = 46 if t=='IDP' else 35 if t=='SAW' else print("WARNING: unknown value for taskname!")
    
    df_list = []
    
    fnames = listdir(datadir)
    
    fnames.sort() # sort IDs such that order during loop is always the same (especially between different OS)
    
    for i,ID in enumerate(fnames):
        if ID not in exclusion_ids:
            rts = []
            n_corr = 0  # no of correct responses
            n_total = 0 # no of responses
            n_missing = 0
            n_tooFast = 0
            
            iddir = join(datadir,ID)
            logfiles = listdir(iddir)
            logfiles = [f for f in logfiles if f in filenames]
            if len(logfiles)>1: print("WARNING: Multiple logfiles for ID "+ID+". First file will be used.")
            logfile = logfiles[0]
            f = join(iddir, logfile)
            with open(f,"r", encoding='utf-8-sig') as read_file:
                # assume json file
                tmp = json.loads(json.load(read_file)['data'])
            for el in tmp:
                if el['sender']=="trial_screen":
                    if el['ended_on']=='response':
                        # exclude RTs < 150ms and missing values
                        if el['duration'] is None: 
                            n_missing += 1
                            # print('Missing data for '+ID+' in '+t+' trial '+el['sender_id']+' [duration]')
                        elif el['duration'] <= 150: n_tooFast += 1
                        else:
                            # exclude responses after deadline was exceeded (IDP: 80s)
                            if t!='IDP' or (sum(rts) + el['duration']) < 80000:
                                n_total += 1
                                if el['correct']==True: n_corr += 1
                                rts.append(el['duration'])
            
            # create row for subject-wise dataframe
            entry = pd.DataFrame()
            entry['ID']         = [ID ]
            entry[t+'_MaxCORR'] = [max_trials ]
            entry[t+'_CORR']    = [n_corr ]
            entry[t+'_ERR']     = [n_total - n_corr ]
            entry[t+'_Missing'] = [n_missing ]
            entry[t+'_TooFast'] = [n_tooFast ]
            entry[t+'_PER']     = [n_corr / max_trials * 100 ]
            entry[t+'_ACC']     = [n_corr / n_total    * 100 ]
            entry[t+'_RT']      = [mean(rts)  / 1000 ]
            entry[t+'_RT_SD']   = [stdev(rts) / 1000 ]
            
            df_list.append(entry)
    
    return pd.concat(df_list, ignore_index=True)


def load_and_format_SWM_data(datadir,filenames, exclusion_ids=[]): 
    df_list = []
    max_trials_SWM = 96
    max_trials_SER = 64
    
    fnames = listdir(datadir)
    
    fnames.sort() # sort IDs such that order during loop is always the same (especially between different OS)
    
    for i,ID in enumerate(fnames):
        if ID not in exclusion_ids:
            rts_SWM = []
            rts_SER = []
            n_corr_SWM = 0  # no of correct responses
            n_corr_SER = 0  # no of correct responses
            n_total_SWM = 0 # no of responses
            n_total_SER = 0 # no of responses
            n_missing_SWM = 0
            n_missing_SER = 0
            n_tooFast_SWM = 0
            n_tooFast_SER = 0
            inverse = False  # indicator if keyboard keys are reversed (if True, 'm' key for 'Yes' response)
    
            practice_finished = False
        
            iddir = join(datadir,ID)
            logfiles = listdir(iddir)
            logfiles = [f for f in logfiles if f in filenames]
            if len(logfiles)>1: print("WARNING: Multiple logfiles for ID "+ID+". First file will be used.")
            logfile = logfiles[0]
            if 'inv' in logfile: inverse = True
            f = join(iddir, logfile)
            with open(f,"r", encoding='utf-8-sig') as read_file:
                # assume json file
                tmp = json.loads(json.load(read_file)['data'])
            for el in tmp:
                if el['sender']=='PRACTICE_END': practice_finished = True
                if practice_finished:
                    if el['sender'] in ["question","question1"]:
                        # QUESTION 1 (SWM)
                        skip_question2 = False
                        if el['ended_on']=='response':
                            # exclude RTs < 150ms and missing values
                            if el['duration'] is None:  
                                n_missing_SWM += 1
                                skip_question2 = True # missing data for question 1
                                # print('Missing data for '+ID+' in SWM trial '+el['sender_id']+' [duration]')
                            elif el['duration'] <= 150: n_tooFast_SWM += 1
                            else:
                                if 'correct' not in el.keys(): 
                                    n_missing_SWM += 1
                                    skip_question2 = True # missing data for question 1
                                    # print('Missing data for '+ID+' in SWM trial '+el['sender_id']+' [correct]')
                                else:
                                    # valid Q1 response
                                    if el['correct']==True: n_corr_SWM += 1
                                    elif (not inverse and el['response']=='m') or (inverse and el['response']=='y'): skip_question2 = True # false positive -> Q2 will be skipped
                                    n_total_SWM += 1
                                    rts_SWM.append(el['duration'])
                        elif el['ended_on']=='timeout':
                            n_total_SWM += 1 # count timeouts as errors
                            skip_question2 = True # Q2 response after Q1 timeout is invalid
    
                    elif "question2" in el['sender']:
                        # QUESTION 2 (SER)
                        if not skip_question2:
                            if el['ended_on']=='response':
                                # exclude RTs < 150ms and missing values
                                if el['duration'] is None:  
                                    n_missing_SER += 1
                                    # print('Missing data for '+ID+' in SER trial '+el['sender_id']+' [correct]')
                                elif el['duration'] <= 150: n_tooFast_SER += 1
                                else:
                                    n_total_SER += 1
                                    if el['correct']==True: n_corr_SER += 1
                                    rts_SER.append(el['duration'])
                            elif el['ended_on']=='timeout':
                                n_total_SER += 1 # count timeouts as errors
                                #print('timeout SER for '+ID+' in trial sender-ID: '+el['sender_id']+' with duration '+str(el['duration']))
    
            # create row for subject-wise dataframe
            entry = pd.DataFrame()
            entry['ID']          = [ID ]
            entry['SWM_MaxCORR'] = [max_trials_SWM ]
            entry['SWM_CORR']    = [n_corr_SWM ]
            entry['SWM_ERR']     = [n_total_SWM - n_corr_SWM ]
            entry['SWM_Missing'] = [n_missing_SWM ]
            entry['SWM_TooFast'] = [n_tooFast_SWM ]
            entry['SWM_PER']     = [n_corr_SWM / max_trials_SWM * 100 ]
            entry['SWM_ACC']     = [n_corr_SWM / n_total_SWM    * 100 ]
            entry['SWM_RT']      = [mean(rts_SWM)  / 1000 ]
            entry['SWM_RT_SD']   = [stdev(rts_SWM) / 1000 ]
            
            entry['SER_MaxCORR'] = [max_trials_SER ]
            entry['SER_CORR']    = [n_corr_SER ]
            entry['SER_ERR']     = [n_total_SER - n_corr_SER ]
            entry['SER_Missing'] = [n_missing_SER ]
            entry['SER_TooFast'] = [n_tooFast_SER ]
            entry['SER_PER']     = [n_corr_SER / max_trials_SER * 100 ]
            entry['SER_ACC']     = [n_corr_SER / n_total_SER    * 100 ]
            if len(rts_SER) > 0:
                entry['SER_RT']      = [mean(rts_SER)  / 1000 ]
            if len(rts_SER) > 1:
                entry['SER_RT_SD']   = [stdev(rts_SER) / 1000 ]
            
            entry['COMP_ACC']    = [ (n_total_SWM * entry['SWM_ACC'].item() + n_total_SER * entry['SER_ACC'].item()) / (n_total_SWM + n_total_SER)]
            if len(rts_SER) > 0:
                entry['COMP_RT'] = [ (n_total_SWM * entry['SWM_RT'].item()  + n_total_SER * entry['SER_RT'].item())  / (n_total_SWM + n_total_SER)]
            
            df_list.append(entry)
            
    return pd.concat(df_list, ignore_index=True)



def sigmoid(x):
    return 1./(1+np.exp(-x))


def convertJSONlogfileToCSV(jsonFilePath, targetdir=None): 
    if path.isfile(jsonFilePath):
        with open(jsonFilePath,"r", encoding='utf-8-sig') as read_file:
            # assume json logfile with big data wrapper
            tmp = json.loads(json.load(read_file)['data'])
        df = pd.DataFrame(tmp)
        if targetdir != None: df.to_csv(path.join(targetdir,path.splitext(path.basename(jsonFilePath))[0]+'.csv'), sep=";")
        return df

def getSWMtrialInfos(taskJSONpath):
    with open(taskJSONpath,"r", encoding='utf-8-sig') as read_file:
        # assume json file
        tmp = json.load(read_file)['components']
    
    df_list = []
    trialNumber = 1
    
    entry = pd.DataFrame()
    
    for i,(c_key,c_value) in enumerate(tmp.items()):
        if c_value['type']=='lab.flow.Sequence' and all(ele.isdigit() for ele in c_value['title'].split('-')):
            # new trial
            # create row for subject-wise dataframe
            entry = pd.DataFrame()
            entry['id'] = [trialNumber ]
            entry['block'] = int(c_value['title'].split('-')[0])
            entry['trial'] = int(c_value['title'].split('-')[1])
            entry['len'] = 4 # default length of sequence
    
            # loop through children of sequence
            children_keys = c_value['children']
            for k in children_keys:
                child = tmp[k]
                if 'blue' in child['title']:
                    # sequential stimulus
                    number = child['title'][4:]
                    if number == '7':  entry['len'] = 7  # long sequence
                    for j,e in enumerate(child['content']):
                        if e["type"] == "circle" and e["fill"]=="#0070d9":
                            entry['pos'+number+'_X'] = [e['left'] ]
                            entry['pos'+number+'_Y'] = [e['top'] ]
                elif child['title'] == 'question' or "question1" in child['title']:
                    # question 1
                    for j,e in enumerate(child['content']):
                        if e["type"] == "circle" and e["fill"]=="#0070d9":
                            entry['Q1_X'] = [e['left'] ]
                            entry['Q1_Y'] = [e['top'] ]
                elif "question2" in child['title']:
                    # question 2
                    for j,e in enumerate(child['content']):
                        if e["type"] == "i-text" and e["fill"]=="#ffffff" and e["text"].isnumeric():
                            entry['Q2_pos'] = [int(e['text']) ]
                            
            # trial end
            # -> infer correct responses
            entry['Q1_corr']=0
            for i in range(entry['len'].item()):
                if entry['pos'+str(i+1)+'_X'].item() == entry['Q1_X'].item() and entry['pos'+str(i+1)+'_Y'].item() == entry['Q1_Y'].item(): 
                    entry['Q1_corr']=1; entry['Q2_corr']=i
            # -> store infos
            df_list.append(entry)
            trialNumber += 1
    
    return pd.concat(df_list, ignore_index=True)

