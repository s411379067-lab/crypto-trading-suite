from math import inf

from pattern_viewer.metrics import calculate_viewer_metrics, completed_trade_pnls


def _order(order_id, side, price, timestamp, qty=1.0):
    return {
        "id": order_id,
        "side": side,
        "status": "filled",
        "fill_price": price,
        "fill_ts": timestamp,
        "created_ts": timestamp,
        "qty": qty,
    }


def test_completed_trade_pnls_combines_entries_and_keeps_breakeven_closes():
    orders = [
        _order("open-1", "long", 100.0, 1, qty=1),
        _order("open-2", "long", 110.0, 2, qty=1),
        _order("close-1", "short", 105.0, 3, qty=1),
        _order("close-2", "short", 105.0, 4, qty=1),
        _order("open-3", "long", 105.0, 5, qty=1),
        _order("close-3", "short", 105.0, 6, qty=1),
    ]

    assert completed_trade_pnls(orders) == [0.0, 0.0, 0.0]


def test_completed_trade_pnls_counts_reverse_fill_remainder_as_new_position():
    orders = [
        _order("open-long", "long", 100.0, 1, qty=1),
        _order("reverse-short", "short", 110.0, 2, qty=2),
        _order("close-short", "long", 100.0, 3, qty=1),
    ]

    assert completed_trade_pnls(orders) == [10.0, 10.0]


def test_filtered_metrics_uses_only_days_with_closed_trades():
    metrics = calculate_viewer_metrics([
        ("2026-10-01", [20.0, -10.0]),
        ("2026-10-01", [10.0]),
        ("2026-10-02", [-20.0]),
        ("2026-10-03", []),
    ])

    assert metrics.trades == 4
    assert metrics.wins == 2
    assert metrics.trading_days == 2
    assert metrics.total_pnl == 0.0
    assert metrics.profit_factor == 1.0
    assert metrics.win_rate == 0.5
    assert metrics.pnl_per_day == 0.0
    assert metrics.pnl_per_trade == 0.0


def test_filtered_metrics_handles_no_losses_and_no_trades():
    positive = calculate_viewer_metrics([("2026-10-01", [5.0, 5.0])])
    empty = calculate_viewer_metrics([("2026-10-01", [])])

    assert positive.profit_factor == inf
    assert positive.win_rate == 1.0
    assert positive.pnl_per_day == 10.0
    assert empty.profit_factor is None
    assert empty.win_rate is None
    assert empty.pnl_per_day is None
