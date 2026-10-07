import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets

from pattern_analyzer.main_window import MainWindow
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
    assert panel.metric_labels["Lots"].text() == "20.0"
    assert panel.metric_labels["Est Loss"].text() == "100.00 USD"
    assert panel.metric_labels["Est Profit"].text() == "200.00 USD"
    assert panel.metric_labels["RR"].text() == "2.00"

    panel.equity_spin.setValue(10_000)
    panel.risk_mode_button.click()
    assert panel.risk_mode_button.text() == "Risk %"
    assert panel.risk_input.value() == 0.5
    assert panel.metric_labels["Risk Target"].text() == "50.00 USD"
    assert panel.metric_labels["Lots"].text() == "10.0"


def test_short_risk_direction_and_cancel_preview():
    app, panel = _panel()
    panel.btn_short.click()
    panel.entry_edit.setText("100")
    panel.sl_edit.setText("105")
    panel.tp_edit.setText("90")

    assert panel.metric_labels["Lots"].text() == "20.0"
    assert panel.metric_labels["Est Profit"].text() == "200.00 USD"
    assert "background:#ef5350" in panel.btn_place.styleSheet()

    panel.btn_cancel_plan.click()
    assert panel.entry_edit.text() == "0"
    assert panel.sl_edit.text() == "0"
    assert panel.tp_edit.text() == "0"
    assert panel.metric_labels["Lots"].text() == "--"


def test_risk_sized_lots_round_down_to_one_decimal():
    app, panel = _panel()
    panel.entry_edit.setText("100")
    panel.sl_edit.setText("93")
    panel.tp_edit.setText("110")

    assert panel.qty_spin.decimals() == 1
    assert panel.qty_spin.singleStep() == 0.1
    assert panel.qty_spin.value() == 14.2
    assert panel.metric_labels["Lots"].text() == "14.2"
    assert panel.metric_labels["Est Loss"].text() == "99.40 USD"


def test_pending_orders_and_order_records_sections_remain_available():
    app, panel = _panel()
    panel.show()
    app.processEvents()

    assert panel.pending_box.isVisible()
    assert panel.records_box.isVisible()
    assert panel.history_splitter.count() == 2
    assert panel.history_splitter.widget(0) is panel.pending_box
    assert panel.history_splitter.widget(1) is panel.records_box
    assert panel.pending_table.columnCount() == 6
    assert panel.records_table.columnCount() == 8
    assert panel.btn_cancel.text() == "Cancel Selected"
    assert panel.btn_delete_record.text() == "Delete Selected Record"
    assert panel.btn_export.text() == "Export Records"
    assert panel.btn_clean.text() == "Clean Records"


def test_main_splitter_panels_fit_screen_and_resize_without_collapsing(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)

    assert window.splitter.count() == 3
    assert all(not window.splitter.isCollapsible(index) for index in range(3))
    assert window.left_splitter.minimumWidth() == 430
    assert window.chart.minimumWidth() == 870
    assert window.research.minimumWidth() == 380
    assert window.chart.toolbar.layout().count() == 3
    assert window.minimumSizeHint().width() < 1920

    window.close()
