<<<<<<< Updated upstream
# Event codes for Memory Guided Search -- TODO: Add attend red/green code, Confirm correctness of yes/no ids
import numpy as np, pandas as pd

trial_ids = ['Stim/S182','Stim/S183'] # no block ids, just trial ids

timelocks = {
    "stim": {'left':'Stim/S 10','right':'Stim/S 11'},
    "prob": {'nomatch':'Stim/S110','match':'Stim/S111'},
    "resp": {'correct':'Stim/S189','incorrect':'Stim/S190','none':'Stim/S191'}
}

intervals = {'stim':(-.2,1.2),
            'prob':(-.2,1.2),
            'resp':(-1.2,.2)}

# prepare event codes for specific conditions
meta_ids = {
    "TO1": {i+1:f'Stim/S{201+i}' for i in range(8)},
    "TO2": {i+1:f'Stim/S{209+i}' for i in range(8)},
    "DO1": {i+1:f'Stim/S{217+i}' for i in range(8)},
    "DO2": {i+1:f'Stim/S{225+i}' for i in range(8)},
    "NO1": {i+1:f'Stim/S{233+i}' for i in range(8)},
    "NO2": {i+1:f'Stim/S{241+i}' for i in range(8)},
    "Cue_side": {'left':'Stim/S 10','right':'Stim/S 11'},
    "TT_match": {0:'Stim/S110',1:'Stim/S111'},
    "DT_match": {0:'Stim/S160',1:'Stim/S161'},
    "TLoc": {i+1:f'Stim/S{121+i}' for i in range(12)}, # Probe target location
    "TOr": {i+1:f'Stim/S{141+i}' for i in range(8)}, # Probe target orientation
    "DOr": {i+1:f'Stim/S{151+i}' for i in range(8)}, # Probe distractor orientation
    "Resp_type": {'yes': 'Stim/S192', 'no': 'Stim/S193','none':'Stim/S194'},
    "Correct": {'incorrect': 'Stim/S190', 'correct': 'Stim/S189'}
}

rt_ids = {'start':[timelocks['prob']['nomatch'],timelocks['prob']['match']], 'end':[timelocks['resp']['incorrect'],timelocks['resp']['correct'],timelocks['resp']['none']]}

def get_metadata(evt,ids,f=None):
    md = {k:[] for k in ["Block","Trial"]+list(meta_ids.keys())+["RT"]}
    # translate task ids to EEG-embedded codes
    meta_codes = {k: {ids[t]:s for s,t in v.items()} for k,v in meta_ids.items()}
    # change to numeric code for correct
    if 'Correct' in meta_codes.keys():
        meta_codes['Correct'] = {ids[meta_ids['Correct']['correct']]:1}

    # Event codes for measuring RT -- both must be unique within a trial
    start_id = [ids[rt] for rt in rt_ids['start']] # list: begin RT codes
    end_id = [ids[rt] for rt in rt_ids['end']] # list: end RT codes

    # separate by blocks
    blocks = [(-1,-1)] # blocks not encoded
    for i,b in enumerate(blocks):
        evt_b = evt[b[0]+1:b[1],:]
        k=1;
        # separate by trials
        trials = list(zip(np.where(evt_b[:,2]==ids[trial_ids[0]])[0],np.where(evt_b[:,2]==ids[trial_ids[1]])[0]))
        for j,t in enumerate(trials):
            evt_t = evt_b[t[0]+1:t[1],:] # <- all the events within a trial format: [[time,_,event_code],...]
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
            if j>0 and md['Cue_side'][-2]!=md['Cue_side'][-1]:
                k+=1
            md['Block'].extend([k])
            begin = sum([evt_t[k,0] for k,e in enumerate(evt_t[:,2]) if e in start_id])
            end = sum([evt_t[k,0] for k,e in enumerate(evt_t[:,2]) if e in end_id])
            rt = end-begin
            if rt<0:
                rt = float('nan')
            md['RT'].extend([rt])
    return pd.DataFrame(md)
=======
# Event codes for Memory Guided Search
from p0ly_utils.metadata.core import (
    CodeLookup,
    ExperimentSpec,
    InferFromColumn,
    IntSum,
    PairedMarkers,
    RTMeasure,
)
from p0ly_utils.metadata.parser import parse_metadata

trial_ids = ["Stim/S182", "Stim/S183"]

timelocks = {
    "stim": {"left": "Stim/S 10", "right": "Stim/S 11"},
    "prob": {"nomatch": "Stim/S110", "match": "Stim/S111"},
    "resp": {"correct": "Stim/S189", "incorrect": "Stim/S190", "none": "Stim/S191"},
}

intervals = {
    "stim": (-0.2, 1.2),
    "prob": (-0.2, 1.2),
    "resp": (-1.2, 0.2),
}

rt_ids = {
    "start": [timelocks["prob"]["nomatch"], timelocks["prob"]["match"]],
    "end": [
        timelocks["resp"]["incorrect"],
        timelocks["resp"]["correct"],
        timelocks["resp"]["none"],
    ],
}

spec = ExperimentSpec(
    name="mgsearch",
    timelocks=timelocks,
    intervals=intervals,
    block_strategy=InferFromColumn("Cue_side", end=-1),
    trial_strategy=PairedMarkers("Stim/S182", "Stim/S183", offset=(1, 0)),
    columns={
        "TO1": IntSum({f"Stim/S{201 + i}": i + 1 for i in range(8)}),
        "TO2": IntSum({f"Stim/S{209 + i}": i + 1 for i in range(8)}),
        "DO1": IntSum({f"Stim/S{217 + i}": i + 1 for i in range(8)}),
        "DO2": IntSum({f"Stim/S{225 + i}": i + 1 for i in range(8)}),
        "NO1": IntSum({f"Stim/S{233 + i}": i + 1 for i in range(8)}),
        "NO2": IntSum({f"Stim/S{241 + i}": i + 1 for i in range(8)}),
        "Cue_side": CodeLookup({"left": "Stim/S 10", "right": "Stim/S 11"}),
        "TT_match": IntSum({"Stim/S110": 0, "Stim/S111": 1}),
        "DT_match": IntSum({"Stim/S160": 0, "Stim/S161": 1}),
        "TLoc": IntSum({f"Stim/S{121 + i}": i + 1 for i in range(12)}),
        "TOr": IntSum({f"Stim/S{141 + i}": i + 1 for i in range(8)}),
        "DOr": IntSum({f"Stim/S{151 + i}": i + 1 for i in range(8)}),
        "Resp_type": CodeLookup(
            {"yes": "Stim/S192", "no": "Stim/S193", "none": "Stim/S194"}
        ),
        "Correct": IntSum({"Stim/S189": 1}),
    },
    rt_defs=[
        RTMeasure(
            "RT",
            start=["Stim/S110", "Stim/S111"],
            end=["Stim/S190", "Stim/S189", "Stim/S191"],
        )
    ],
)


def get_metadata(evt, ids, f=None):
    return parse_metadata(spec, evt, ids, csv_path=f)
>>>>>>> Stashed changes
