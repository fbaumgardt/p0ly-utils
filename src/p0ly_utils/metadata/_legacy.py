"""Legacy get_metadata implementations kept for migration validation."""

import numpy as np
import pandas as pd


def dotprobe_get_metadata(evt, ids, f=None):
    trial_ids = ["Stim/S  9", "Stim/S 10"]
    meta_ids = {
        "Cue_type": {"salient": "Stim/S 11", "mix": "Stim/S 12", "neutral": "Stim/S 13"},
        "Cue_format": {"TN": "Stim/S 16", "NT": "Stim/S 17"},
        "Dot_type": {"vertical": "Stim/S 43", "horizontal": "Stim/S 44"},
        "Dot_side": {"left": "Stim/S 45", "right": "Stim/S 46"},
        "Resp_type": {"top": "Stim/S 37", "bottom": "Stim/S 38"},
        "Correct": {"incorrect": "Stim/S 31", "correct": "Stim/S 34"},
    }
    rt_ids = {"start": ["Stim/S 14"], "end": ["Stim/S 30"]}

    md = {k: [] for k in ["Block", "Trial"] + list(meta_ids.keys()) + ["RT"]}
    meta_codes = {k: {ids[t]: s for s, t in v.items()} for k, v in meta_ids.items()}
    if "Correct" in meta_codes.keys():
        meta_codes["Correct"] = {ids[meta_ids["Correct"]["correct"]]: True}

    start_id = [ids[rt] for rt in rt_ids["start"]]
    end_id = [ids[rt] for rt in rt_ids["end"]]

    blocks = [(0, -1)]
    for i, b in enumerate(blocks):
        evt_b = evt
        trials = list(
            zip(
                np.where(evt_b[:, 2] == ids[trial_ids[0]])[0],
                np.where(evt_b[:, 2] == ids[trial_ids[1]])[0],
            )
        )
        for j, t in enumerate(trials):
            evt_t = evt_b[t[0] : t[1], :]
            md["Block"].extend([i + 1])
            md["Trial"].extend([j + 1])
            for col, code in meta_codes.items():
                match list(code.values())[0]:
                    case str():
                        md[col].extend(["".join([code.get(e, "") for e in evt_t[:, 2]])])
                    case bool():
                        md[col].extend([any([code.get(e, False) for e in evt_t[:, 2]])])
                    case int():
                        md[col].extend([sum([code.get(e, 0) for e in evt_t[:, 2]])])
                    case _:
                        md[col].extend([[code.get(e) for e in evt_t[:, 2] if e in code.keys()]])
            begin = sum([evt_t[k, 0] for k, e in enumerate(evt_t[:, 2]) if e in start_id])
            end = sum([evt_t[k, 0] for k, e in enumerate(evt_t[:, 2]) if e in end_id])
            rt = end - begin
            if rt < 0:
                rt = float("nan")
            md["RT"].extend([rt])
    return pd.DataFrame(md)


def mgsearch_get_metadata(evt, ids, f=None):
    trial_ids = ["Stim/S182", "Stim/S183"]
    meta_ids = {
        "TO1": {i + 1: f"Stim/S{201 + i}" for i in range(8)},
        "TO2": {i + 1: f"Stim/S{209 + i}" for i in range(8)},
        "DO1": {i + 1: f"Stim/S{217 + i}" for i in range(8)},
        "DO2": {i + 1: f"Stim/S{225 + i}" for i in range(8)},
        "NO1": {i + 1: f"Stim/S{233 + i}" for i in range(8)},
        "NO2": {i + 1: f"Stim/S{241 + i}" for i in range(8)},
        "Cue_side": {"left": "Stim/S 10", "right": "Stim/S 11"},
        "TT_match": {0: "Stim/S110", 1: "Stim/S111"},
        "DT_match": {0: "Stim/S160", 1: "Stim/S161"},
        "TLoc": {i + 1: f"Stim/S{121 + i}" for i in range(12)},
        "TOr": {i + 1: f"Stim/S{141 + i}" for i in range(8)},
        "DOr": {i + 1: f"Stim/S{151 + i}" for i in range(8)},
        "Resp_type": {"yes": "Stim/S192", "no": "Stim/S193", "none": "Stim/S194"},
        "Correct": {"incorrect": "Stim/S190", "correct": "Stim/S189"},
    }
    rt_ids = {
        "start": ["Stim/S110", "Stim/S111"],
        "end": ["Stim/S190", "Stim/S189", "Stim/S191"],
    }

    md = {k: [] for k in ["Block", "Trial"] + list(meta_ids.keys()) + ["RT"]}
    meta_codes = {k: {ids[t]: s for s, t in v.items()} for k, v in meta_ids.items()}
    if "Correct" in meta_codes.keys():
        meta_codes["Correct"] = {ids[meta_ids["Correct"]["correct"]]: 1}

    start_id = [ids[rt] for rt in rt_ids["start"]]
    end_id = [ids[rt] for rt in rt_ids["end"]]

    blocks = [(-1, -1)]
    for i, b in enumerate(blocks):
        evt_b = evt[b[0] + 1 : b[1], :]
        k = 1
        trials = list(
            zip(
                np.where(evt_b[:, 2] == ids[trial_ids[0]])[0],
                np.where(evt_b[:, 2] == ids[trial_ids[1]])[0],
            )
        )
        for j, t in enumerate(trials):
            evt_t = evt_b[t[0] + 1 : t[1], :]
            md["Trial"].extend([j + 1])
            for col, code in meta_codes.items():
                match list(code.values())[0]:
                    case str():
                        md[col].extend(["".join([code.get(e, "") for e in evt_t[:, 2]])])
                    case bool():
                        md[col].extend([any([code.get(e, False) for e in evt_t[:, 2]])])
                    case int():
                        md[col].extend([sum([code.get(e, 0) for e in evt_t[:, 2]])])
                    case _:
                        md[col].extend([[code.get(e) for e in evt_t[:, 2] if e in code.keys()]])
            if j > 0 and md["Cue_side"][-2] != md["Cue_side"][-1]:
                k += 1
            md["Block"].extend([k])
            begin = sum([evt_t[k, 0] for k, e in enumerate(evt_t[:, 2]) if e in start_id])
            end = sum([evt_t[k, 0] for k, e in enumerate(evt_t[:, 2]) if e in end_id])
            rt = end - begin
            if rt < 0:
                rt = float("nan")
            md["RT"].extend([rt])
    return pd.DataFrame(md)


def dmss_get_metadata(evt, ids, f=None):
    block_ids = ["Stim/S  3", "Stim/S  4"]
    trial_ids = ["Stim/S  5", "Stim/S  6"]
    probe_id = "Stim/S 57"
    stimulus_id = {"size1": "Stim/S 11", "size2": "Stim/S 25", "size4": "Stim/S 43"}
    response_id = {"all": "Stim/S 60", "correct": "Stim/S 64", "incorrect": "Stim/S 65"}

    md = {"Block": [], "Trial": [], "Size": [], "RT": [], "Correct": []}
    size_id = {
        ids[stimulus_id["size1"]]: 1,
        ids[stimulus_id["size2"]]: 2,
        ids[stimulus_id["size4"]]: 4,
    }
    correct_id = {ids[response_id["correct"]]: 1}
    blocks = list(
        zip(
            np.where(evt[:, 2] == ids[block_ids[0]])[0],
            np.where(evt[:, 2] == ids[block_ids[1]])[0],
        )
    )
    start_id = ids[probe_id]
    end_id = ids[response_id["all"]]
    for i, b in enumerate(blocks):
        evt_b = evt[b[0] + 1 : b[1], :]
        trials = list(
            zip(
                np.where(evt_b[:, 2] == ids[trial_ids[0]])[0],
                np.where(evt_b[:, 2] == ids[trial_ids[1]])[0],
            )
        )
        for j, t in enumerate(trials):
            evt_t = evt_b[t[0] + 1 : t[1], :]
            md["Block"].extend([i + 1])
            md["Trial"].extend([j + 1])
            md["Correct"].extend([sum([correct_id.get(e, 0) for e in evt_t[:, 2]])])
            md["Size"].extend([sum([size_id.get(e, 0) for e in evt_t[:, 2]])])
            begin = sum([evt_t[k, 0] for k, e in enumerate(evt_t[:, 2]) if e == start_id])
            end = sum([evt_t[k, 0] for k, e in enumerate(evt_t[:, 2]) if e == end_id])
            md["RT"].extend([end - begin])
    return pd.DataFrame(md)
