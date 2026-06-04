# Event codes for Memory Guided Search
from p0ly_utils.metadata.core import (
    CodeLookup,
    ExperimentSpec,
    IntSum,
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

spec = ExperimentSpec(
    name="mgsearch",
    timelocks=timelocks,
    intervals=intervals,
    trial_codes=["Stim/S182"],
    infer_block_from="Cue_side",
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


def get_metadata(df, f=None):
    return parse_metadata(spec, df, csv_path=f)
