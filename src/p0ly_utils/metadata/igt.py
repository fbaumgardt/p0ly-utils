from p0ly_utils.metadata.core import (
    ExpandOnEvent,
    ExperimentSpec,
    IntSum,
    ListCollect,
    RTMeasure,
)
from p0ly_utils.metadata.parser import parse_metadata

block_ids = ["Stim/S 20", "Stim/S 21"]
trial_ids = ["Stim/S 30", "Stim/S 31"]
select_id = {"select": "Stim/S 40"}
submit_id = {"submit": "Stim/S 50"}
feedback_id = {"feedback": "Stim/S 60"}
card_id = {"A": "Stim/S 41", "B": "Stim/S 42", "C": "Stim/S 43", "D": "Stim/S 44"}
deck_id = {"1": "Stim/S 45", "2": "Stim/S 46", "3": "Stim/S 47", "4": "Stim/S 48"}
win_id = {"win": "Stim/S 61", "loss": "Stim/S 62", "zero": "Stim/S 63"}

timelocks = {
    "select": {"all": "Stim/S 40"},
    "submit": {"all": "Stim/S 50"},
    "fdb": {"all": "Stim/S 60"},
}

intervals = {
    "select": (-1.2, 0.2),
    "submit": (-1.2, 0.2),
    "fdb": (-0.2, 1.2),
}

spec = ExperimentSpec(
    name="igt",
    timelocks=timelocks,
    intervals=intervals,
    block_codes=["Stim/S 20"],
    trial_codes=["Stim/S 30"],
    #trial_end_codes=["Stim/S 31"],
    columns={
        "Result": IntSum({"Stim/S 61": 1}),
    },
    rt_defs=[RTMeasure("RT_Submit", start=["Stim/S 30"], end=["Stim/S 50"])],
    trial_expander=ExpandOnEvent(
        event_code="Stim/S 40",
        per_event_columns={
            "Card": ListCollect(
                {"A": "Stim/S 41", "B": "Stim/S 42", "C": "Stim/S 43", "D": "Stim/S 44"}
            ),
            "Deck": ListCollect(
                {"1": "Stim/S 45", "2": "Stim/S 46", "3": "Stim/S 47", "4": "Stim/S 48"}
            ),
        },
    ),
)


def get_metadata(df, f=None, sel_trials=False):
    return parse_metadata(spec, df, csv_path=f, expand_trials=sel_trials)
