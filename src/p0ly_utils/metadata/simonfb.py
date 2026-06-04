# Event codes for SimonFB
from p0ly_utils.metadata.core import (
    BoolPresence,
    CodeLookup,
    DerivedColumn,
    ExperimentSpec,
    RTMeasure,
)
from p0ly_utils.metadata.parser import parse_metadata

stimulus_id = {"all": "Stim/S 12"}
response_id = {"all": "Stim/S 30", "none": "Stim/S 39"}
feedback_id = {"all": "Stim/S 22"}
timelocks = {
    "stim": stimulus_id,
    "resp": response_id,
    "fdb": feedback_id,
}

intervals = {
    "stim": (-0.2, 1),
    "resp": (-1, 0.2),
    "fdb": (-0.2, 1),
}

BLOCKSIZE = 60

spec = ExperimentSpec(
    name="simonfb",
    timelocks=timelocks,
    intervals=intervals,
    block_codes=["Stim/S  7"],
    trial_codes=["Stim/S 17"],
    columns={
        "Color": CodeLookup(
            {
                "red": "Stim/S 16",
                "purple": "Stim/S 45",
                "blue": "Stim/S 46",
                "yellow": "Stim/S 19",
            }
        ),
        "Side": CodeLookup({"left": "Stim/S 15", "right": "Stim/S 14"}),
        "Response": CodeLookup({"left": "Stim/S 37", "right": "Stim/S 38"}),
        "Correct": BoolPresence("Stim/S 34"),
        "Congruent": DerivedColumn(
            depends_on=["Side", "Response", "Correct"],
            fn=lambda row: (row["Side"] == row["Response"]) == row["Correct"],
        ),
        "Block": DerivedColumn(
            depends_on=["Trial"],
            fn=lambda row: (row["Trial"] - 1) // BLOCKSIZE,
        ),
    },
    rt_defs=[RTMeasure("RT", start=["Stim/S 12"], end=["Stim/S 30"])],
)


def get_metadata(df, f=None):
    return parse_metadata(spec, df, csv_path=f)
