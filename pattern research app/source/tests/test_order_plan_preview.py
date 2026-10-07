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

    assert len(panel.pending_orders) == 1
    record = panel.pending_orders[0]
    assert record["side"] == "long"
    assert record["order_type"] == "limit"
    assert record["requested_price"] == 100.0
    assert record["qty"] == 20.0
    assert record["stop_loss"] == 95.0
    assert record["take_profit"] == 110.0
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

    assert panel.pending_orders == []
    assert len(panel.case.orders) == 1
    record = panel.case.orders[0]
    assert record["side"] == "short"
    assert record["order_type"] == "market"
    assert record["requested_price"] is None
    assert record["fill_price"] == 100.0
    assert record["qty"] == 20.0
    assert record["stop_loss"] == 105.0
    assert record["take_profit"] == 90.0
    assert changed and fills
    assert "filled at 100" in panel.status_label.text()


def test_send_rejects_invalid_bracket_without_creating_order():
    app = _app()
    panel = OrderPanel()
    _attach_replay(panel)
    panel.sl_edit.setText("105")

    panel.btn_place.click()

    assert panel.pending_orders == []
    assert panel.case.orders == []
    assert "SL must be below Entry" in panel.status_label.text()


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
    assert "BUY LIMIT" in entry_html and "Lots 20.00" in entry_html
    assert "SL 95.00" in sl_html and "100.00 USD" in sl_html
    assert "TP 110.00" in tp_html and "200.00 USD" in tp_html and "2.00R" in tp_html

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
