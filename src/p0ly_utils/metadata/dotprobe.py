# Event codes for Dot Probe -- READY FOR TESTING
import numpy as np, pandas as pd

block_ids = ['Stim/S103']
trial_ids = ['Stim/S  9','Stim/S 10']

timelocks = {
    "dots": {'all':'Stim/S 41'},
    "cue": {'all':'Stim/S 14'},
    "resp": {'yes':'Stim/S 30'}
}

intervals = {
    'dots': (.2,1),
    'cue': (.2,1),
    'resp': (-1.2,.2)
}

# event codes for specific conditions
meta_ids = {
    "Cue_type": {'salient':'Stim/S 11','mix':'Stim/S 12','neutral':'Stim/S 13'},
    "Cue_format": {'TN':'Stim/S 16','NT':'Stim/S 17'},
    "Dot_type": {'vertical':'Stim/S 43','horizontal':'Stim/S 44'},
    "Dot_side": {'left':'Stim/S 45','right':'Stim/S 46'},
    "Resp_type": {'top': 'Stim/S 37', 'bottom': 'Stim/S 38'},
    "Correct": {'incorrect': 'Stim/S 31', 'correct': 'Stim/S 34'}
}

rt_ids = {'start':[timelocks['cue']['all']], 'end':[timelocks['resp']['yes']]}

def get_metadata(evt,ids,f=None):
    md = {k:[] for k in ["Block","Trial"]+list(meta_ids.keys())+["RT"]}
    # translate task ids to EEG-embedded codes
    meta_codes = {k: {ids[t]:s for s,t in v.items()} for k,v in meta_ids.items()}
    # change to numeric code for correct
    if 'Correct' in meta_codes.keys():
        meta_codes['Correct'] = {ids[meta_ids['Correct']['correct']]:True}

    # Event codes for measuring RT
    start_id = [ids[rt] for rt in rt_ids['start']] # list: begin RT codes
    end_id = [ids[rt] for rt in rt_ids['end']] # list: end RT codes

    # separate by blocks -- acount for missing block_start events
    block_evts = np.append([0],np.where(evt[:,2]==ids[block_ids[0]])[0])
    blocks = list(zip(block_evts[:-1],block_evts[1:]))
    blocks=[(0,-1)]
    for i,b in enumerate(blocks):
        evt_b = evt#[b[0]:b[1],:]
        # separate by trials
        trials = list(zip(np.where(evt_b[:,2]==ids[trial_ids[0]])[0],np.where(evt_b[:,2]==ids[trial_ids[1]])[0]))
        for j,t in enumerate(trials):
            evt_t = evt_b[t[0]:t[1],:] # <- all the events within a trial format: [[time,_,event_code],...]
            md['Block'].extend([i+1])
            md['Trial'].extend([j+1])
            for col,code in meta_codes.items():
                match list(code.values())[0]:
                    case str():
                        md[col].extend(["".join([code.get(e,'') for e in evt_t[:,2]])])
                    case bool():
                        md[col].extend([any([code.get(e,False) for e in evt_t[:,2]])])
                    case int():
                        md[col].extend([sum([code.get(e,0) for e in evt_t[:,2]])])
                    case _:
                        md[col].extend([[code.get(e) for e in evt_t[:,2] if e in code.keys()]])
            begin = sum([evt_t[k,0] for k,e in enumerate(evt_t[:,2]) if e in start_id])
            end = sum([evt_t[k,0] for k,e in enumerate(evt_t[:,2]) if e in end_id])
            rt = end-begin
            if rt<0:
                rt = float('nan')
            md['RT'].extend([rt])
    return pd.DataFrame(md)