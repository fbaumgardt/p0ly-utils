from p0ly_utils.metadata.core import (
    ExperimentSpec,
    IntSum,
)
from p0ly_utils.metadata.parser import parse_metadata

trial_ids = [f"Stim/S {t}" for t in range(81, 100)]
cue_id = {"cue/left": "Stim/S201", "cue/right": "Stim/S202"}
response_id = {"resp/incorrect": "Stim/S250", "resp/correct": "Stim/S251"}

timelocks = {
    "cue": cue_id,
    "resp": response_id,
}

intervals = {
    "cue": (-0.2, 1.0),
    "resp": (-1, 0.2),
}

spec = ExperimentSpec(
    name="intwm",
    timelocks=timelocks,
    intervals=intervals,
    trial_codes=trial_ids,
    columns={
        "Correct": IntSum({"Stim/S251": 1, "Stim/S250": 0}),
    },
    csv_columns={
        "Condition": (
            "condArray",
            lambda x: {1: "static", 2: "ignore", 3: "constant", 4: "changing"}.get(x),
        ),
        "Target": ("tarLoci", lambda x: {1: "left", 2: "right"}.get(x)),
        "OrientL": ("Ori_1", lambda x: x),
        "OrientR": ("Ori_2", lambda x: x),
        "OrientT": ("tarOri", lambda x: x),
        "ChgBegin": ("startTime4int", lambda x: x),
        "ChgRT": ("chgRT", lambda x: x),
    },
)


def get_metadata(df, f=None):
    return parse_metadata(spec, df, csv_path=f)
