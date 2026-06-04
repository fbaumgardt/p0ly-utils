# Event codes for DMSS
from p0ly_utils.metadata.core import (
    ExperimentSpec,
    IntSum,
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
    block_codes=["Stim/S  3"],
    trial_codes=["Stim/S  5"],
    #trial_end_codes=["Stim/S  6"],
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


def get_metadata(df, f=None):
    return parse_metadata(spec, df, csv_path=f)
