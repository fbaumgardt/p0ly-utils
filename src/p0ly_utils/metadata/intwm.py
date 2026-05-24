import numpy as np
import pandas as pd

# Event codes for intWM --- WORK IN PROGRESS, READ METADATA FROM CSV

trial_ids = [f'Stim/S {t}' for t in range(81,100)]
#translate_sid = lambda sid: f"stim/T{int(sid[-3:])//100}/L{sid[-2]}/R{sid[-1]}"
#stimulus_id = {translate_sid(s):s for s in [f"Stim/{h*100+z*10+d:3.0f}" for h in [0,1] for z in range(1,7) for d in range(1,7)]}
#translate_dots = lambda dots: f"dots/C{dots[-2]}/D{dots[-1]}"
#dots_id =  {translate_dots(s):s for s in [f"Stim/{200+z*10+d:3.0f}" for h in [0,1] for z in range(1,5) for d in range(1,7)]}
cue_id =  {'cue/left': 'Stim/S201', 'cue/right': 'Stim/S202'}
response_id =  {'resp/incorrect': 'Stim/S250', 'resp/correct': 'Stim/S251'}

timelocks = {
    #'stim': stimulus_id,
    #'dots': dots_id,
    'cue': cue_id,
    'resp': response_id
}

intervals = {
    #'stim':(-.2,1.),
    #'dots':(-.2,1.),
    'cue':(-.2,1.),
    'resp':(-1,.2)
}

def get_metadata(evt,ids,f=None):
    md = {"Trial":[],"Rep":[],"Target":[],"OrientL":[],"OrientR":[],"OrientDots":[],"DotsType":[],"Cue":[],"Correct":[]}
    # prepare event codes for specific conditions
    
    # list all response ids for correct answers
    correct_id = {ids[response_id['resp/correct']]:1} 

    # Event codes for measuring RT
    #start_id = ids[probe_id['all']] # begin measuring RT
    #end_id = ids[response_id['all']] # end measuring RT

    # separate by blocks
    blocks = [(0,-1)] # intWM event markers don't capture block structure - it's possible to deduce it later
    for i,b in enumerate(blocks):
        evt_b = evt[b[0]:-1,:]
        # separate by trials
        trial_codes = [ids[t] for t in trial_ids]
        trials = list(zip(np.where([e in trial_codes for e in evt_b[:,2]])[0],np.where(evt_b[:,2]==ids[trial_ids[1]])[0]))
        for j,t in enumerate(trials):
            evt_t = evt_b[t[0]+1:t[1],:] # <- all the events within a trial format: [[time,_,event_code],...]
            md['Trial'].extend([j+1])
            md['Correct'].extend([sum([correct_id.get(e,0) for e in evt_t[:,2]])])
    return pd.DataFrame(md)

    # NEW STUFF
def nothing():
    md = pd.read_csv("intWM.csv")
    columns = {'Condition': {'condArray':lambda x: {1:'static',2:'ignore',3:'constant',4:'changing'}.get(x)},
          'Target': {'tarLoci': lambda x: {1:'left',2:'right'}.get(x)},
          'OrientL': {'Ori_1': lambda x: x},
          'OrientR': {'Ori_2': lambda x: x},
          'OrientT': {'tarOri': lambda x: x},
           'ChgBegin': {'startTime4int': lambda x: x},
           'ChgRT': {'chgRT': lambda x: x}
           #'ColorL': {1:'',2:'',3:''}.get(x)},
           #'ColorR': {1:'',2:'',3:''}.get(x)},
    }
    return (columns,md)