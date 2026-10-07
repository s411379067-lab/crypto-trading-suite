import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets

from pattern_analyzer.order_panel import OrderPanel


def _panel():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = OrderPanel()
    return app, panel


def test_panel_starts_in_pending_long_mode():
    app, panel = _panel()

    assert panel.selected_side == "long"
    assert panel.mode_pending.isChecked()
    assert panel.risk_mode_button.text() == "Risk $"
    assert panel.risk_input.value() == 100.0
    assert panel.btn_place.text() == "SEND"
    assert panel.btn_place.isEnabled()
    assert panel.order_type_label.text() == "Order Type: --"


def test_panel_mode_and_direction_buttons_update_order_selection():
    app, panel = _panel()

    panel.btn_short.click()
    assert panel.selected_side == "short"
    assert panel.btn_short.isChecked()

    panel.mode_market.click()
    assert panel.mode_market.isChecked()
    assert not panel.entry_edit.isEnabled()
    assert panel.order_type_label.text() == "Order Type: market"

    panel.mode_pending.click()
    assert panel.mode_pending.isChecked()
    assert panel.entry_edit.isEnabled()


def test_risk_cash_and_percent_modes_recalculate_lots_and_metrics():
    app, panel = _panel()
    panel.entry_edit.setText("100")
    panel.sl_edit.setText("95")
    panel.tp_edit.setText("110")

    assert panel.metric_labels["Risk Target"].text() == "100.00 USD"
    assert panel.metric_labels["Lots"].text() == "20.000000"
    assert panel.metric_labels["Est Loss"].text() == "100.00 USD"
    assert panel.metric_labels["Est Profit"].text() == "200.00 USD"
    assert panel.metric_labels["RR"].text() == "2.00"

    panel.equity_spin.setValue(10_000)
    panel.risk_mode_button.click()
    assert panel.risk_mode_button.text() == "Risk %"
    assert panel.risk_input.value() == 0.5
    assert panel.metric_labels["Risk Target"].text() == "50.00 USD"
    assert panel.metric_labels["Lots"].text() == "10.000000"


def test_short_risk_direction_and_cancel_preview():
    app, panel = _panel()
    panel.btn_short.click()
    panel.entry_edit.setText("100")
    panel.sl_edit.setText("105")
    panel.tp_edit.setText("90")

    assert panel.metric_labels["Lots"].text() == "20.000000"
    assert panel.metric_labels["Est Profit"].text() == "200.00 USD"
    assert "background:#ef5350" in panel.btn_place.styleSheet()

    panel.btn_cancel_plan.click()
    assert panel.entry_edit.text() == "0"
    assert panel.sl_edit.text() == "0"
    assert panel.tp_edit.text() == "0"
    assert panel.metric_labels["Lots"].text() == "--"
