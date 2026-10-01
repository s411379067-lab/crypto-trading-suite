from shared_core.ohlc import candle_change_pct, format_ohlc_text


def test_candle_change_pct_uses_close_vs_open():
    assert candle_change_pct(25000, 25050) == 0.2
    assert candle_change_pct(25000, 24950) == -0.2


def test_candle_change_pct_handles_zero_open():
    assert candle_change_pct(0, 100) is None


def test_format_ohlc_includes_signed_change_and_forming_state():
    row = {
        "open": 25000,
        "high": 25100,
        "low": 24900,
        "close": 25050,
        "is_partial": True,
    }
    text = format_ohlc_text(row)
    assert "開=25000.00" in text
    assert "高=25100.00" in text
    assert "低=24900.00" in text
    assert "收=25050.00" in text
    assert "漲跌=+0.200%" in text
    assert "[forming]" in text


def test_format_ohlc_placeholder():
    assert format_ohlc_text(None) == "開=--  高=--  低=--  收=--  漲跌=--"
