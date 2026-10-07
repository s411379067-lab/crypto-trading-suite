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
    assert panel.type_combo.currentText() == "limit"
    assert panel.type_combo.isEnabled()
    assert panel.btn_place.text() == "Send Pending"


def test_panel_mode_and_direction_buttons_update_order_selection():
    app, panel = _panel()

    panel.btn_short.click()
    assert panel.selected_side == "short"
    assert panel.btn_short.isChecked()

    panel.mode_market.click()
    assert panel.mode_market.isChecked()
    assert panel.type_combo.currentText() == "market"
    assert not panel.type_combo.isEnabled()
    assert panel.btn_place.text() == "Send Market"

    panel.mode_pending.click()
    assert panel.mode_pending.isChecked()
    assert panel.type_combo.currentText() == "limit"
    assert panel.type_combo.isEnabled()
    assert panel.btn_place.text() == "Send Pending"

