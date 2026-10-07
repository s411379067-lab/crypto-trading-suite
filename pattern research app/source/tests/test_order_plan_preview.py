import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from pyqtgraph.Qt import QtWidgets

from pattern_analyzer.chart_widget import ChartWidget
from pattern_analyzer.order_panel import OrderPanel


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _attach_replay(panel):
    panel.case = SimpleNamespace(orders=[], display={"timezone": "UTC"}, touch=lambda: None)
    panel.raw_df = pd.DataFrame([
        {"timestamp": 1_000.0, "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0},
    ])
    panel.replay = SimpleNamespace(current_ts=1_000.0)
    panel.entry_edit.setText("100")
    panel.sl_edit.setText("95")
    panel.tp_edit.setText("110")


def test_panel_emits_transient_plan_and_accepts_chart_price_edits():
    app = _app()
    panel = OrderPanel()
    plans = []
    panel.order_plan_changed.connect(plans.append)

    panel.entry_edit.setText("100")
    panel.sl_edit.setText("95")
    panel.tp_edit.setText("110")

    assert plans[-1]["mode"] == "pending"
    assert plans[-1]["side"] == "long"
    assert plans[-1]["entry"] == 100.0
    assert plans[-1]["sl"] == 95.0
    assert plans[-1]["tp"] == 110.0

    panel.set_plan_price_from_chart("sl", 94.5)
    assert panel.sl_edit.text() == "94.50"

    panel.set_order_mode("market")
    panel.set_plan_price_from_chart("entry", 99)
    assert panel.entry_edit.text() != "99.0000"


def test_send_creates_pending_order_with_bracket_and_risk_size():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)

    panel.btn_place.click()

    assert len(panel.pending_orders) == 3
    record = next(order for order in panel.pending_orders if not order.get("role"))
    assert record["side"] == "long"
    assert record["order_type"] == "limit"
    assert record["requested_price"] == 100.0
    assert record["qty"] == 20.0
    assert record["stop_loss"] == 95.0
    assert record["take_profit"] == 110.0
    protections = [order for order in panel.pending_orders if order.get("role")]
    assert {order["role"] for order in protections} == {"stop_loss", "take_profit"}
    assert all(not order["armed"] for order in protections)
    assert all(order["parent_order_id"] == record["id"] for order in protections)
    assert all(panel.pending_table.item(row, 5).text() == "WAIT ENTRY" for row in (1, 2))
    assert panel.case.orders == []
    assert "order sent" in panel.status_label.text()


def test_send_market_fills_at_replay_price_and_emits_updates():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    panel.set_side("short")
    panel.sl_edit.setText("105")
    panel.tp_edit.setText("90")
    panel.set_order_mode("market")
    changed = []
    fills = []
    panel.changed.connect(lambda: changed.append(True))
    panel.fills_changed.connect(lambda: fills.append(True))

    panel.btn_place.click()

    assert len(panel.pending_orders) == 2
    assert len(panel.case.orders) == 1
    record = panel.case.orders[0]
    assert record["side"] == "short"
    assert record["order_type"] == "market"
    assert record["requested_price"] is None
    assert record["fill_price"] == 100.0
    assert record["qty"] == 20.0
    assert record["stop_loss"] == 105.0
    assert record["take_profit"] == 90.0
    assert all(order["armed"] for order in panel.pending_orders)
    assert all(panel.pending_table.item(row, 5).text() == "ACTIVE" for row in (0, 1))
    assert changed and fills
    assert "filled at 100" in panel.status_label.text()


def test_send_uses_risk_sized_lots_rounded_down_to_one_decimal():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    panel.sl_edit.setText("93")

    assert panel.qty_spin.value() == 14.2
    assert panel.submit_plan()
    entry = next(order for order in panel.pending_orders if not order.get("role"))
    protections = [order for order in panel.pending_orders if order.get("role")]
    assert entry["qty"] == 14.2
    assert all(order["qty"] == 14.2 for order in protections)


def test_multiple_tp_plan_splits_lots_previews_lines_and_blocks_unsafe_send():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    chart = ChartWidget()
    chart.replay = object()
    panel.order_plan_changed.connect(chart.set_order_plan)
    chart.order_plan_price_changed.connect(panel.set_plan_price_from_chart)

    assert panel.add_take_profit_target()
    assert [row["qty"].value() for row in panel._take_profit_rows] == [10.0, 10.0]
    panel._take_profit_rows[1]["price"].setText("120")

    assert panel.metric_labels["Est Profit"].text() == "300.00 USD"
    assert panel.metric_labels["RR"].text() == "3.00"
    assert "Runner: 0.0" in panel.tp_allocation_label.text()
    assert set(chart.order_plan_items) == {"entry", "sl", "tp", "tp:tp-2"}
    assert "TP2 120.00" in chart.order_plan_labels["tp:tp-2"].textItem.toHtml()
    assert "Lots 10.0" in chart.order_plan_labels["tp:tp-2"].textItem.toHtml()

    chart.order_plan_price_changed.emit("tp:tp-2", 125.0)
    assert panel._take_profit_rows[1]["price"].text() == "125.00"
    assert not panel.submit_plan()
    assert panel.case.orders == []
    assert panel.pending_orders == []
    assert "next order stage" in panel.status_label.text()


def test_removing_tp_target_returns_its_lots_to_remaining_target():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    assert panel.add_take_profit_target()
    second_id = panel._take_profit_rows[1]["id"]

    assert panel.remove_take_profit_target(second_id)

    assert len(panel._take_profit_rows) == 1
    assert panel.tp_lots_spin.value() == 20.0
    assert "Runner: 0.0" in panel.tp_allocation_label.text()


def test_tp_lots_cannot_exceed_position_and_remaining_is_shown_as_runner():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    assert panel.add_take_profit_target()

    panel._take_profit_rows[0]["qty"].setValue(15.0)
    assert [row["qty"].value() for row in panel._take_profit_rows] == [10.0, 10.0]
    panel._take_profit_rows[0]["qty"].setValue(7.0)
    assert [row["qty"].value() for row in panel._take_profit_rows] == [7.0, 10.0]
    assert "Runner: 3.0" in panel.tp_allocation_label.text()


def test_tenths_allocation_can_reassign_remaining_runner_to_any_tp():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    panel.risk_input.setValue(3.0)
    assert panel.add_take_profit_target()
    assert panel.add_take_profit_target()
    rows = panel._take_profit_rows

    rows[1]["qty"].setValue(0.2)
    rows[2]["qty"].setValue(0.0)
    rows[0]["qty"].setValue(0.3)
    assert [row["qty"].value() for row in rows] == [0.3, 0.2, 0.0]
    assert "Runner: 0.1" in panel.tp_allocation_label.text()

    rows[2]["qty"].setValue(0.1)
    assert [row["qty"].value() for row in rows] == [0.3, 0.2, 0.1]
    assert "Runner: 0.0" in panel.tp_allocation_label.text()

    rows[2]["qty"].setValue(0.0)
    rows[0]["qty"].setValue(0.4)
    assert [row["qty"].value() for row in rows] == [0.4, 0.2, 0.0]


def test_send_rejects_invalid_bracket_without_creating_order():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    panel.sl_edit.setText("105")

    panel.btn_place.click()

    assert panel.pending_orders == []
    assert panel.case.orders == []
    assert "SL must be below Entry" in panel.status_label.text()


def test_pending_entry_arms_protection_only_after_entry_fill():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    panel.entry_edit.setText("99")
    assert panel.submit_plan()
    entry = next(order for order in panel.pending_orders if not order.get("role"))

    panel.replay.current_ts = 1_060.0
    panel.raw_df = pd.concat([panel.raw_df, pd.DataFrame([
        {"timestamp": 1_060.0, "open": 100.0, "high": 101.0, "low": 98.0, "close": 100.0},
    ])], ignore_index=True)
    panel.process_replay_advance(1_000.0, 1_060.0)

    assert entry["status"] == "filled"
    protections = [order for order in panel.pending_orders if order.get("role")]
    assert len(protections) == 2
    assert all(order["armed"] for order in protections)
    assert all(order["active_from_ts"] == 1_060.0 for order in protections)


def test_cancelling_pending_entry_removes_its_waiting_protection_pair():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    assert panel.submit_plan()
    entry = next(order for order in panel.pending_orders if not order.get("role"))
    panel.pending_table.selectRow(0)

    panel.cancel_selected()

    assert panel.pending_orders == []
    assert panel._active_entry_id is None


def test_confirm_and_cancel_bracket_edits_apply_or_restore_prices():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    plans = []
    panel.order_plan_changed.connect(plans.append)
    panel.set_order_mode("market")
    assert panel.submit_plan()
    entry = panel.case.orders[0]
    panel.sl_edit.setText("94")
    panel.tp_edit.setText("112")

    assert panel.confirm_bracket_update()
    assert entry["stop_loss"] == 94.0
    assert entry["take_profit"] == 112.0
    assert {order["requested_price"] for order in panel.pending_orders} == {94.0, 112.0}
    assert plans[-1]["bracket_dirty"] is False
    assert panel.cancel_bracket_update() is False

    panel.sl_edit.setText("93")
    assert plans[-1]["bracket_dirty"] is True
    assert panel.cancel_bracket_update()
    assert panel.sl_edit.text() == "94.00"
    assert panel.tp_edit.text() == "112.00"


def test_active_protection_orders_are_rebuilt_when_case_is_reopened():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    panel.set_order_mode("market")
    assert panel.submit_plan()
    case = panel.case
    raw_df = panel.raw_df
    replay = panel.replay

    reopened = OrderPanel()
    reopened.set_context(case, raw_df, replay)

    assert reopened._active_entry_id == case.orders[0]["id"]
    assert len(reopened.pending_orders) == 2
    assert all(order["armed"] for order in reopened.pending_orders)


def test_take_profit_trigger_closes_trade_and_cancels_stop_loss_sibling():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    panel.set_order_mode("market")
    assert panel.submit_plan()
    parent = panel.case.orders[0]

    panel.raw_df = pd.concat([panel.raw_df, pd.DataFrame([
        {"timestamp": 1_060.0, "open": 100.0, "high": 111.0, "low": 99.0, "close": 110.0},
    ])], ignore_index=True)
    panel.replay.current_ts = 1_060.0
    panel.process_replay_advance(1_000.0, 1_060.0)

    assert parent["bracket_status"] == "closed"
    assert len(panel.case.orders) == 2
    close = panel.case.orders[1]
    assert close["order_type"] == "take profit"
    assert close["fill_price"] == 110.0
    assert close["computed_action"] == "CLOSE"
    assert panel.pending_orders == []
    assert panel._position is None


def test_full_close_preserves_exact_legacy_position_quantity():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    panel.case.orders = [{
        "id": "legacy-entry", "status": "filled", "side": "long",
        "fill_ts": 1_000.0, "created_ts": 1_000.0, "fill_price": 100.0,
        "qty": 1.25,
    }]

    panel.close_position()

    close = panel.case.orders[-1]
    assert close["origin"] == "manual-close"
    assert close["qty"] == 1.25
    assert panel._position is None


def test_manual_close_cancels_active_protection_orders():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    panel.case.orders = [{
        "id": "live-entry", "status": "filled", "side": "long",
        "fill_ts": 1_000.0, "created_ts": 1_000.0, "fill_price": 100.0,
        "qty": 0.6, "stop_loss": 95.0, "take_profit": 110.0,
        "bracket_status": "active", "bracket_order_ids": ["sl-live", "tp-live"],
    }]
    panel._active_entry_id = "live-entry"
    panel._confirmed_bracket = (95.0, 110.0)
    panel.pending_orders = [
        {"id": "sl-live", "parent_order_id": "live-entry", "role": "stop_loss", "status": "open"},
        {"id": "tp-live", "parent_order_id": "live-entry", "role": "take_profit", "status": "open"},
    ]

    panel.close_position()

    assert panel._position is None
    assert panel.pending_orders == []
    assert panel._active_entry_id is None
    assert panel.case.orders[0]["bracket_status"] == "closed"
    assert panel.case.orders[-1]["origin"] == "manual-close"


def test_live_entry_chart_exposes_close_x_but_pending_entry_does_not():
    app = _app()
    chart = ChartWidget()
    chart.replay = object()
    chart.set_order_plan({"mode": "pending", "entry": 100.0, "sl": 95.0, "tp": 110.0})
    assert "close" not in chart.order_plan_actions

    requested = []
    chart.position_close_requested.connect(lambda: requested.append(True))
    chart.set_order_plan({
        "mode": "market", "entry": 100.0, "sl": 95.0, "tp": 110.0,
        "bracket_edit_enabled": True,
    })
    assert "close" in chart.order_plan_actions
    chart.order_plan_actions["close"].clicked.emit("close")
    app.processEvents()
    assert requested == [True]


def test_panel_buttons_confirm_or_cancel_live_bracket_edits():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    entry = {
        "id": "live-entry", "status": "filled", "side": "long",
        "fill_ts": 1_000.0, "created_ts": 1_000.0, "fill_price": 100.0,
        "qty": 1.0, "stop_loss": 95.0, "take_profit": 110.0,
        "bracket_status": "active",
    }
    panel.case.orders = [entry]
    panel._active_entry_id = entry["id"]
    panel._confirmed_bracket = (95.0, 110.0)
    panel.pending_orders = [
        {"id": "sl-live", "parent_order_id": entry["id"], "role": "stop_loss", "requested_price": 95.0},
        {"id": "tp-live", "parent_order_id": entry["id"], "role": "take_profit", "requested_price": 110.0},
    ]
    panel.refresh()

    assert panel.btn_place.text() == "CONFIRM"
    assert not panel.btn_place.isEnabled()
    assert not panel.btn_cancel_plan.isEnabled()
    panel.sl_edit.setText("94")
    assert panel.btn_place.isEnabled()
    assert panel.btn_cancel_plan.isEnabled()
    assert "#8b5cf6" in panel.btn_place.styleSheet()
    assert "#8b5cf6" in panel.btn_cancel_plan.styleSheet()

    panel.btn_place.click()
    assert entry["stop_loss"] == 94.0
    assert panel.pending_orders[0]["requested_price"] == 94.0
    assert not panel.btn_place.isEnabled()

    panel.tp_edit.setText("112")
    assert panel.btn_cancel_plan.isEnabled()
    panel.btn_cancel_plan.click()
    assert panel.tp_edit.text() == "110.00"
    assert not panel.btn_cancel_plan.isEnabled()


def test_trade_metrics_show_realized_and_unrealized_pnl_with_direction_colors():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    panel.case.orders = [
        {"id": "entry", "status": "filled", "side": "long", "fill_ts": 990.0,
         "created_ts": 990.0, "fill_price": 100.0, "qty": 1.0},
        {"id": "partial-close", "status": "filled", "side": "short", "fill_ts": 995.0,
         "created_ts": 995.0, "fill_price": 105.0, "qty": 0.4},
    ]
    panel.raw_df.loc[0, "close"] = 102.0
    panel.refresh()

    assert panel.metric_labels["Realized PnL"].text() == "+$2.00"
    assert panel.metric_labels["Unrealized PnL"].text() == "+$1.20"
    assert "#26a69a" in panel.metric_labels["Realized PnL"].styleSheet()
    assert "#26a69a" in panel.metric_labels["Unrealized PnL"].styleSheet()
    assert panel.metrics_separator.frameShape() == QtWidgets.QFrame.HLine

    panel.raw_df.loc[0, "close"] = 98.0
    panel.refresh()
    assert panel.metric_labels["Unrealized PnL"].text() == "-$1.20"
    assert "#ef5350" in panel.metric_labels["Unrealized PnL"].styleSheet()


def test_chart_draws_and_updates_transient_order_plan_lines():
    app = _app()
    chart = ChartWidget()
    chart.replay = object()
    changed = []
    chart.order_plan_price_changed.connect(lambda field, price: changed.append((field, price)))

    chart.set_order_plan({
        "mode": "pending",
        "side": "long",
        "entry": 100.0,
        "sl": 95.0,
        "tp": 110.0,
        "order_type": "limit",
        "lots": 20.0,
        "est_loss": 100.0,
        "est_profit": 200.0,
        "rr": 2.0,
    })

    assert set(chart.order_plan_items) == {"entry", "sl", "tp"}
    assert chart.ORDER_PLAN_LABEL_POSITION == 0.985
    assert chart.order_plan_items["entry"].value() == 100.0
    assert chart.order_plan_items["sl"].value() == 95.0
    assert chart.order_plan_items["tp"].value() == 110.0
    entry_html = chart.order_plan_labels["entry"].textItem.toHtml()
    sl_html = chart.order_plan_labels["sl"].textItem.toHtml()
    tp_html = chart.order_plan_labels["tp"].textItem.toHtml()
    assert "BUY LIMIT" in entry_html and "Lots 20.0" in entry_html
    assert "SL 95.00" in sl_html and "100.00 USD" in sl_html
    assert "TP 110.00" in tp_html and "200.00 USD" in tp_html and "2.00R" in tp_html

    chart.set_order_plan({"mode": "market", "entry": 100, "sl": 95, "tp": 110,
                          "bracket_edit_enabled": True, "bracket_dirty": False})
    assert set(chart.order_plan_actions) == {"close"}
    chart.set_order_plan({"mode": "market", "entry": 100, "sl": 94, "tp": 112,
                          "bracket_edit_enabled": True, "bracket_dirty": True})
    chart.resize(900, 600)
    chart.show()
    app.processEvents()
    chart._position_order_plan_labels()
    pixel_size = chart.plot.getViewBox().viewPixelSize()[0]
    entry_label = chart.order_plan_labels["entry"]
    entry_left = entry_label.pos().x() - entry_label.textItem.boundingRect().width() * pixel_size
    close = chart.order_plan_actions["close"]
    assert pixel_size > 0
    assert abs(close.pos().x() - (entry_left - 6.0 * pixel_size)) < 1e-6

    line = chart.order_plan_items["sl"]
    line.setValue(94.0)
    chart._order_plan_line_finished("sl", line)
    assert changed == [("sl", 94.0)]

    chart.set_order_plan({})
    assert chart.order_plan_items == {}


def test_panel_and_chart_plan_stay_synchronized_in_both_directions():
    app = _app()
    panel = OrderPanel()
    chart = ChartWidget()
    chart.replay = object()
    panel.order_plan_changed.connect(chart.set_order_plan)
    chart.order_plan_price_changed.connect(panel.set_plan_price_from_chart)

    panel.entry_edit.setText("120")
    panel.sl_edit.setText("115")
    panel.tp_edit.setText("130")
    assert chart.order_plan_items["entry"].value() == 120.0
    assert chart.order_plan_items["sl"].value() == 115.0
    assert chart.order_plan_items["tp"].value() == 130.0

    chart.order_plan_price_changed.emit("sl", 114.0)
    assert panel.sl_edit.text() == "114.00"
    assert chart.order_plan_items["sl"].value() == 114.0

    panel.set_order_mode("market")
    chart.set_order_plan({"mode": "market", "entry": 120.0, "sl": 115.0, "tp": 130.0})
    assert chart.order_plan_items["entry"].movable is False


def test_pending_type_is_derived_from_side_entry_and_replay_price():
    app = _app()
    panel = OrderPanel()
    panel.raw_df = pd.DataFrame([{"timestamp": 1.0, "close": 100.0}])
    panel.replay = SimpleNamespace(current_ts=1.0)

    panel.entry_edit.setText("99")
    assert panel.order_type_label.text() == "Order Type: limit"
    panel.entry_edit.setText("101")
    assert panel.order_type_label.text() == "Order Type: stop market"
    panel.set_side("short")
    panel.entry_edit.setText("101")
    assert panel.order_type_label.text() == "Order Type: limit"
    panel.entry_edit.setText("99")
    assert panel.order_type_label.text() == "Order Type: stop market"


def test_market_entry_tracks_replay_price_and_is_locked():
    app = _app()
    panel = OrderPanel()
    panel.raw_df = pd.DataFrame([{"timestamp": 1.0, "close": 100.0}])
    panel.replay = SimpleNamespace(current_ts=1.0)
    chart = ChartWidget()
    chart.replay = object()
    panel.order_plan_changed.connect(chart.set_order_plan)

    panel.set_order_mode("market")
    assert panel.entry_edit.text() == "100.00"
    assert not panel.entry_edit.isEnabled()
    assert chart.order_plan_items["entry"].value() == 100.0
    assert chart.order_plan_items["entry"].movable is False
