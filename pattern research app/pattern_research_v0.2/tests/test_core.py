import pandas as pd
from shared_core.aggregation import aggregate_visible_bars


def make_m1():
    dt = pd.date_range("2026-09-17 13:30:00+00:00", periods=6, freq="1min")
    return pd.DataFrame({
        "dt_utc": dt,
        "timestamp": dt.astype("int64") / 1_000_000_000,
        "open": [100,101,102,103,104,105],
        "high": [101,102,103,104,105,106],
        "low": [99,100,101,102,103,104],
        "close": [101,102,103,104,105,106],
    })


def test_partial_m5():
    df = make_m1()
    current = float(df.iloc[2]["timestamp"])
    start = float(df.iloc[0]["timestamp"])
    out = aggregate_visible_bars(df, "M5", current, start)
    assert len(out) == 1
    assert out.iloc[0]["open"] == 100
    assert out.iloc[0]["high"] == 103
    assert out.iloc[0]["low"] == 99
    assert out.iloc[0]["close"] == 103
    assert bool(out.iloc[0]["is_partial"]) is True


if __name__ == "__main__":
    test_partial_m5()
    print("core test passed")
