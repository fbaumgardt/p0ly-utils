# Event codes for Dot Probe
from p0ly_utils.metadata.core import (
    BoolPresence,
    CodeLookup,
    ExperimentSpec,
    RTMeasure,
)
from p0ly_utils.metadata.parser import parse_metadata

trial_ids = ["Stim/S  9", "Stim/S 10"]

timelocks = {
    "dots": {"all": "Stim/S 41"},
    "cue": {"all": "Stim/S 14"},
    "resp": {"yes": "Stim/S 30"},
}

intervals = {
    "dots": (0.2, 1),
    "cue": (0.2, 1),
    "resp": (-1.2, 0.2),
}

spec = ExperimentSpec(
    name="dotprobe",
    timelocks=timelocks,
    intervals=intervals,
    block_codes=["Stim/S103"],
    trial_codes=["Stim/S  9"],
    columns={
        "Cue_type": CodeLookup(
            {"salient": "Stim/S 11", "mix": "Stim/S 12", "neutral": "Stim/S 13"}
        ),
        "Cue_format": CodeLookup({"TN": "Stim/S 16", "NT": "Stim/S 17"}),
        "Dot_type": CodeLookup({"vertical": "Stim/S 43", "horizontal": "Stim/S 44"}),
        "Dot_side": CodeLookup({"left": "Stim/S 45", "right": "Stim/S 46"}),
        "Resp_type": CodeLookup({"top": "Stim/S 37", "bottom": "Stim/S 38"}),
        "Correct": BoolPresence("Stim/S 34"),
    },
    rt_defs=[RTMeasure("RT", start=["Stim/S 14"], end=["Stim/S 30"])],
)


def get_metadata(df, f=None):
    return parse_metadata(spec, df, csv_path=f)
