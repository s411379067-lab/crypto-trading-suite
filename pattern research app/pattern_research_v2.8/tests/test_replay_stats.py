import pandas as pd
from shared_core.replay_stats import summarize_current_replay_range

def _raw():
    return pd.DataFrame([
        {"timestamp":100.0,"open":100.0,"high":102.0,"low":99.0,"close":101.0},
        {"timestamp":160.0,"open":101.0,"high":104.0,"low":100.0,"close":103.0},
        {"timestamp":220.0,"open":103.0,"high":103.5,"low":97.0,"close":98.0},
        {"timestamp":280.0,"open":98.0,"high":105.0,"low":96.0,"close":104.0},
    ])

def test_current_range_replay_start_to_current_inclusive():
    out=summarize_current_replay_range(_raw(),160.0,220.0)
    assert out["bar_count"]==2 and out["base_open"]==101.0
    assert out["high"]==104.0 and out["low"]==97.0 and out["range_points"]==7.0
    assert abs(out["range_pct"]-(7/101*100))<1e-12

def test_current_range_updates_when_replay_moves_backward():
    assert summarize_current_replay_range(_raw(),100.0,280.0)["range_points"]==9.0
    assert summarize_current_replay_range(_raw(),100.0,160.0)["range_points"]==5.0

def test_current_range_none_for_invalid_window():
    assert summarize_current_replay_range(_raw(),220.0,160.0) is None
    assert summarize_current_replay_range(_raw(),1000.0,1060.0) is None
