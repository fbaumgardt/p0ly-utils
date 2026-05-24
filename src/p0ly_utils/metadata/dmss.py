<<<<<<< Updated upstream
import numpy as np, pandas as pd

block_ids = ['Stim/S  3','Stim/S  4']
trial_ids = ['Stim/S  5','Stim/S  6']
probe_id =  'Stim/S 57'
stimulus_id = {'size1':'Stim/S 11','size2':'Stim/S 25','size4':'Stim/S 43'}
response_id =  {'all':'Stim/S 60','correct': 'Stim/S 64', 'incorrect': 'Stim/S 65'}
feedback_id = {'size1':'Stim/S 72','size2':'Stim/S 73','size4':'Stim/S 74'}

timelocks = {
    "stim": {'size1':'Stim/S 11','size2':'Stim/S 25','size4':'Stim/S 43'},
    "prob": {'all':'Stim/S 57'},
    "resp": {'correct':'Stim/S 64','incorrect':'Stim/S 65'},
    "fdb": {'size1':'Stim/S 72','size2':'Stim/S 73','size4':'Stim/S 74'}
}

intervals = {'stim':(-.2,1.2),
            'prob':(-.2,1.2),
            "fdb":(-.2,1.),
            'resp':(-1.2,.2)}

def get_metadata(evt,ids,f=None):
    md = {"Block":[],"Trial":[],"Size":[],"RT":[],"Correct":[]}
    size_id = {ids[stimulus_id['size1']]:1,ids[stimulus_id['size2']]:2,ids[stimulus_id['size4']]:4}
    correct_id = {ids[response_id['correct']]:1}
    blocks = list(zip(np.where(evt[:,2]==ids[block_ids[0]])[0],np.where(evt[:,2]==ids[block_ids[1]])[0]))
    start_id = ids[probe_id]
    end_id = ids[response_id['all']]
    for i,b in enumerate(blocks):
        evt_b = evt[b[0]+1:b[1],:]
        trials = list(zip(np.where(evt_b[:,2]==ids[trial_ids[0]])[0],np.where(evt_b[:,2]==ids[trial_ids[1]])[0]))
        for j,t in enumerate(trials):
            evt_t = evt_b[t[0]+1:t[1],:]
            md['Block'].extend([i+1])
            md['Trial'].extend([j+1])
            md['Correct'].extend([sum([correct_id.get(e,0) for e in evt_t[:,2]])])
            md['Size'].extend([sum([size_id.get(e,0) for e in evt_t[:,2]])])
            begin = sum([evt_t[k,0] for k,e in enumerate(evt_t[:,2]) if e==start_id])
            end = sum([evt_t[k,0] for k,e in enumerate(evt_t[:,2]) if e==end_id])
            md['RT'].extend([end-begin])
    return pd.DataFrame(md)

=======
# Event codes for DMSS
from p0ly_utils.metadata.core import (
    ExperimentSpec,
    IntSum,
    PairedMarkers,
    RTMeasure,
)
from p0ly_utils.metadata.parser import parse_metadata

block_ids = ["Stim/S  3", "Stim/S  4"]
trial_ids = ["Stim/S  5", "Stim/S  6"]
probe_id = "Stim/S 57"
stimulus_id = {"size1": "Stim/S 11", "size2": "Stim/S 25", "size4": "Stim/S 43"}
response_id = {"all": "Stim/S 60", "correct": "Stim/S 64", "incorrect": "Stim/S 65"}
feedback_id = {"size1": "Stim/S 72", "size2": "Stim/S 73", "size4": "Stim/S 74"}

timelocks = {
    "stim": {"size1": "Stim/S 11", "size2": "Stim/S 25", "size4": "Stim/S 43"},
    "prob": {"all": "Stim/S 57"},
    "resp": {"correct": "Stim/S 64", "incorrect": "Stim/S 65"},
    "fdb": {"size1": "Stim/S 72", "size2": "Stim/S 73", "size4": "Stim/S 74"},
}

intervals = {
    "stim": (-0.2, 1.2),
    "prob": (-0.2, 1.2),
    "fdb": (-0.2, 1.0),
    "resp": (-1.2, 0.2),
}

spec = ExperimentSpec(
    name="dmss",
    timelocks=timelocks,
    intervals=intervals,
    block_strategy=PairedMarkers("Stim/S  3", "Stim/S  4"),
    trial_strategy=PairedMarkers("Stim/S  5", "Stim/S  6"),
    columns={
        "Size": IntSum({"Stim/S 11": 1, "Stim/S 25": 2, "Stim/S 43": 4}),
        "Correct": IntSum({"Stim/S 64": 1}),
    },
    rt_defs=[
        RTMeasure(
            "RT",
            start=["Stim/S 57"],
            end=["Stim/S 60"],
            nan_if_negative=False,
        )
    ],
)


def get_metadata(evt, ids, f=None):
    return parse_metadata(spec, evt, ids, csv_path=f)
>>>>>>> Stashed changes
