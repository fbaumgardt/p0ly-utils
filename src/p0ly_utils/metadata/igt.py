import numpy as np
import pandas as pd

# Event codes for IGT  -- READY FOR TESTING

block_ids = ['Stim/S 20','Stim/S 21']
trial_ids = ['Stim/S 30','Stim/S 31']
select_id =  {'select':'Stim/S 40'}
submit_id =  {'submit': 'Stim/S 50'}
feedback_id =  {'feedback': 'Stim/S 60'}
card_id =  {'A': 'Stim/S 41','B': 'Stim/S 42','C': 'Stim/S 43','D': 'Stim/S 44'}
deck_id =  {'1': 'Stim/S 45','2': 'Stim/S 46','3': 'Stim/S 47','4': 'Stim/S 48'}
win_id =  {'win': 'Stim/S 61','loss': 'Stim/S 62','zero': 'Stim/S 63'}

timelocks = {'select':'Stim/S 40',
            'submit':'Stim/S 50',
            'fdb':'Stim/S 60'}



intervals = {'select':(-1.2,.2),
            'submit':(-1.2,.2),
            'fdb':(-.2,1.2)}


def get_metadata(evt,ids,f=None,sel_trials=False): # sel_trials flag for select-locked analysis, where multiple locks per trial may occur
    md = {"Block":[],"Trial":[],"RT_Select":[],"RT_Submit":[],"Card":[],"Deck":[],"Num_Sel":[],"Total_Sel":[],"Result":[]}
    # prepare event codes for specific conditions
    card_codes = {ids[v]:k for k,v in card_id.items()}
    deck_codes = {ids[v]:k for k,v in deck_id.items()}
    win_codes = {ids[v]:k for k,v in win_id.items()}
     # Stuff to help legacy code pass linter
    correct_id = {card_codes.get(win_id['win'],'a'):1}
    correct_id = {deck_codes.get(win_id['win'],'a'):1} 
    correct_id = {win_codes.get(win_id['win'],'a'):1}


    # Event codes for measuring RT
    start_id = ids[trial_ids[0]] # begin measuring RT
    sel_end_id = ids[select_id['select']] # end measuring select RT
    sub_end_id = ids[submit_id['submit']] # end measuring submit RT

    # separate by blocks
    blocks = list(zip(np.where(evt[:,2]==ids[block_ids[0]])[0],np.where(evt[:,2]==ids[block_ids[1]])[0]))
    for i,b in enumerate(blocks):
        evt_b = evt[b[0]+1:b[1],:]
        # separate by trials
        trials = list(zip(np.where(evt_b[:,2]==ids[trial_ids[0]])[0],np.where(np.sum([evt_b[:,2]==ids[t] for t in trial_ids[1]]))[0]))
        for j,t in enumerate(trials):
            evt_t = evt_b[t[0]:t[1]+1,:] # <- all the events within a trial format: [[time,_,event_code],...]
            sel_events = np.where(evt[:,2]==sel_end_id)[0] # finding the selection events
            card_events = [card_id.get(e) for e in evt_t[:,2] if card_id.get(e,False)] # finding the selection events
            deck_events = [deck_id.get(e) for e in evt_t[:,2] if deck_id.get(e,False)] # finding the selection events
            num_sels = len(sel_events)
            if not sel_trials:
                sel_events = sel_events[-1:]
                card_events = card_events[-1:]
                deck_events = deck_events[-1:]
            for k,s,c,d in zip(range(len(sel_events)),sel_events,card_events,deck_events):
                md['Block'].extend([i+1])
                md['Trial'].extend([j+1])
                begin = sum([evt_t[k,0] for k,e in enumerate(evt_t[:,2]) if e==start_id])
                end = evt_t[s,0]
                md['RT_Select'].extend([end-begin])
                end = sum([evt_t[k,0] for k,e in enumerate(evt_t[:,2]) if e==sub_end_id])
                md['RT_Submit'].extend([end-begin])
                md['Card'].extend([c])
                md['Deck'].extend([d])
                md['Num_Sel'].extend([k])
                md['Total_Sel'].extend([num_sels])
                md['Result'].extend([sum([correct_id.get(e,0) for e in evt_t[:,2]])])
    return pd.DataFrame(md)