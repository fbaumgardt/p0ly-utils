# Event codes for SimonFB -- READY FOR TESTING
import numpy as np, pandas as pd

# for timelocks
stimulus_id = {'all':'Stim/S 12'}
response_id =  {'all':'Stim/S 30','none': 'Stim/S 39'}
feedback_id = {'all':'Stim/S 22'}
timelocks = {"stim":stimulus_id,
            "resp":response_id,
            "fdb":feedback_id}

intervals = {"stim":(-.2,1),
            "resp":(-1,.2),
            "fdb":(-.2,1)}

# for metadata
block_ids = ['Stim/S  7','Stim/S  8']
trial_ids = ['Stim/S 17','Stim/S 18']
fdb_correct_id = {'correct':'Stim/S 23','incorrect':'Stim/S 24'}
resp_correct_id = {'incorrect': 'Stim/S 31','correct': 'Stim/S 34'}
resp_side_id = {'left': 'Stim/S 37', 'right': 'Stim/S 38'}
stim_color_id = {'red':'Stim/S 16','purple':'Stim/S 45','blue':'Stim/S 46','yellow':'Stim/S 19'}
stim_side_id = {'left':'Stim/S 15','right':'Stim/S 14'}
Blocksize = 60

def get_metadata(evt,ids,f=None):
    # evts is a 2-d numpy with a row for every event, containing event time at [0] and id at [2]
    # ids maps the 'Stim/XXX' task-level identifiers to the numeric EEG-embedded ids
    md = {"Block":[],"Trial":[],"Side":[],"Color":[],"Congruent":[],"Response":[],"RT":[],"Correct":[]}
    # prepare event codes for specific conditions
    stim_color_code = {ids[stim_color_id[s]]:s for s in stim_color_id.keys()}
    stim_side_code = {ids[stim_side_id[s]]:s for s in stim_side_id.keys()}
    resp_side_code = {ids[resp_side_id[s]]:s for s in resp_side_id.keys()}
    # list all response ids for correct answers
    correct_id = {ids[resp_correct_id['correct']]:1}

    # Event codes for measuring RT
    start_id = ids[stimulus_id['all']] # begin measuring RT
    end_id = ids[response_id['all']] # end measuring RT

    # separate by blocks
    blocks = list(zip(np.where(evt[:,2]==ids[block_ids[0]])[0],np.where(evt[:,2]==ids[block_ids[1]])[0]))
    for i,b in enumerate(blocks):
        evt_b = evt[b[0]+1:b[1],:]
        # separate by trials
        trials = list(zip(np.where(evt_b[:,2]==ids[trial_ids[0]])[0], np.where(evt_b[:,2]==ids[trial_ids[1]])[0]))
        for j,t in enumerate(trials):
            evt_t = evt_b[t[0]+1:t[1]+1,:] # <- all the events within a trial format: [[time,_,event_code],...]
            md['Block'].extend([j//Blocksize]) # no block markers, so using fixed block size
            md['Trial'].extend([j+1])
            md['Color'].extend(["".join([stim_color_code.get(e,'') for e in evt_t[:,2]])])
            sd = "".join([stim_side_code.get(e,'') for e in evt_t[:,2]])
            rsp = "".join([resp_side_code.get(e,'') for e in evt_t[:,2]])
            same = sd==rsp
            cor = sum([correct_id.get(e,0) for e in evt_t[:,2]])>0
            md['Side'].extend([sd])
            md['Response'].extend([rsp])
            md['Congruent'].extend([same==cor]) # congruent means the correct button side is same as stimulus side
            begin = sum([evt_t[k,0] for k,e in enumerate(evt_t[:,2]) if e==start_id])
            end = sum([evt_t[k,0] for k,e in enumerate(evt_t[:,2]) if e==end_id])
            rt = end-begin
            if rt<0:
                rt = np.nan
            md['RT'].extend([rt])
            md['Correct'].extend([cor])
    return pd.DataFrame(md)