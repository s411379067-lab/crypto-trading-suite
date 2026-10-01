from shared_core.order_overlay import build_order_overlay


def _fill(order_id, side, timestamp, price, qty=1):
    return {"id": order_id, "status": "filled", "side": side, "fill_ts": timestamp, "fill_price": price, "qty": qty}


def test_order_overlay_draws_profit_and_loss_segments_for_completed_positions():
    events, segments = build_order_overlay([
        _fill("open-long", "long", 10, 100),
        _fill("close-long", "short", 20, 103),
        _fill("open-short", "short", 30, 110),
        _fill("close-short", "long", 40, 112),
    ], 40)

    assert len(events) == 4
    assert [(segment["entry_price"], segment["exit_price"], segment["pnl"]) for segment in segments] == [
        (100.0, 103.0, 3.0),
        (110.0, 112.0, -2.0),
    ]


def test_order_overlay_uses_weighted_entry_for_add_and_hides_future_fills():
    events, segments = build_order_overlay([
        _fill("open", "long", 10, 100, 1),
        _fill("add", "long", 20, 104, 1),
        _fill("close", "short", 30, 105, 1),
        _fill("future", "short", 40, 110, 1),
    ], 30)

    assert [event["id"] for event in events] == ["open", "add", "close"]
    assert len(segments) == 1
    assert segments[0]["entry_price"] == 102.0
    assert segments[0]["pnl"] == 3.0
