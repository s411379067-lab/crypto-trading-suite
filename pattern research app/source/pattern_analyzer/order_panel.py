from __future__ import annotations

import uuid
from datetime import datetime, timezone
import pandas as pd
from pyqtgraph.Qt import QtCore, QtWidgets
from shared_core.order_overlay import build_order_overlay


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _side_mult(side: str) -> int:
    return 1 if side == "long" else -1


class OrderPanel(QtWidgets.QWidget):
    """Legacy-style order simulator backed by ResearchCase.orders event records."""

    changed = QtCore.Signal()
    fills_changed = QtCore.Signal()

    R_VALUE = 60.0

    @staticmethod
    def _style_order_table(table: QtWidgets.QTableWidget) -> None:
        """Keep order tables readable on the dark application theme."""
        header = table.horizontalHeader()
        header.setDefaultAlignment(QtCore.Qt.AlignCenter)
        header.setMinimumHeight(30)
        font = header.font()
        font.setBold(True)
        font.setPointSize(max(font.pointSize(), 10))
        header.setFont(font)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.case = None
        self.raw_df = pd.DataFrame()
        self.replay = None
        self._position = None
        self._realized_pnl = 0.0
        self._realized_r = 0.0
        # Pending orders are intentionally session-only.
        # Only actual fills are persisted to ResearchCase.orders.
        self.pending_orders: list[dict] = []

        self.setStyleSheet(
            "QLabel { color:#9aa9bf; }"
            "QLineEdit, QDoubleSpinBox { background:#111722; color:#eef3fb; border:1px solid #303b4d; padding:4px; }"
            "QPushButton { background:#202836; color:#9aa9bf; border:1px solid #343e50; padding:6px; }"
            "QPushButton:checked { color:white; font-weight:700; }"
        )
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        self.type_combo = QtWidgets.QComboBox(); self.type_combo.addItems(["market", "limit", "stop market"])
        self.price_edit = QtWidgets.QLineEdit(); self.price_edit.setPlaceholderText("Entry")
        self.qty_spin = QtWidgets.QDoubleSpinBox()
        self.qty_spin.setDecimals(6); self.qty_spin.setRange(0.0, 1_000_000.0); self.qty_spin.setValue(0.0)
        self.type_combo.hide(); self.price_edit.hide(); self.qty_spin.hide()

        self._selected_side = "long"
        self._risk_mode = "cash"
        self._risk_values = {"cash": 100.0, "percent": 0.5}

        top = QtWidgets.QHBoxLayout()
        input_box = QtWidgets.QWidget()
        input_grid = QtWidgets.QGridLayout(input_box)
        input_grid.setContentsMargins(0, 0, 0, 0)
        input_grid.setHorizontalSpacing(6)
        input_grid.setVerticalSpacing(4)
        self.equity_spin = QtWidgets.QDoubleSpinBox()
        self.equity_spin.setDecimals(2); self.equity_spin.setRange(0.0, 1_000_000_000_000.0)
        self.equity_spin.setValue(0.0); self.equity_spin.setGroupSeparatorShown(True)
        self.risk_mode_button = QtWidgets.QPushButton("Risk $")
        self.risk_mode_button.setFixedWidth(72)
        self.risk_input = QtWidgets.QDoubleSpinBox()
        self.risk_input.setDecimals(2); self.risk_input.setRange(0.0, 1_000_000_000.0)
        self.risk_input.setValue(100.0); self.risk_input.setGroupSeparatorShown(True)
        self.entry_edit = QtWidgets.QLineEdit("0")
        self.sl_edit = QtWidgets.QLineEdit("0")
        self.tp_edit = QtWidgets.QLineEdit("0")
        for row, (name, widget) in enumerate((
            ("Equity", self.equity_spin), ("", self.risk_input), ("Entry", self.entry_edit),
            ("SL", self.sl_edit), ("TP", self.tp_edit),
        )):
            if row == 1:
                input_grid.addWidget(self.risk_mode_button, row, 0)
                input_grid.addWidget(widget, row, 1)
            else:
                input_grid.addWidget(QtWidgets.QLabel(name), row, 0)
                input_grid.addWidget(widget, row, 1)
        top.addWidget(input_box, 1)

        metrics_box = QtWidgets.QGroupBox("TRADE METRICS")
        metrics_grid = QtWidgets.QGridLayout(metrics_box)
        self.metric_labels = {}
        for row, key in enumerate(("Est Loss", "Est Profit", "RR", "Risk Target", "Lots")):
            value = QtWidgets.QLabel("--")
            value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            metrics_grid.addWidget(QtWidgets.QLabel(key), row, 0)
            metrics_grid.addWidget(value, row, 1)
            self.metric_labels[key] = value
        top.addWidget(metrics_box, 1)
        outer.addLayout(top)

        self.order_type_label = QtWidgets.QLabel("Order Type: --")
        self.order_type_label.setStyleSheet("background:#111722; border:1px solid #303b4d; padding:6px;")
        outer.addWidget(self.order_type_label)

        self.mode_pending = QtWidgets.QPushButton("PENDING")
        self.mode_market = QtWidgets.QPushButton("MARKET")
        self.btn_short = QtWidgets.QPushButton("SHORT")
        self.btn_long = QtWidgets.QPushButton("LONG")
        for button in (self.mode_pending, self.mode_market, self.btn_short, self.btn_long):
            button.setCheckable(True)
        self.mode_group = QtWidgets.QButtonGroup(self); self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.mode_pending); self.mode_group.addButton(self.mode_market)
        self.side_group = QtWidgets.QButtonGroup(self); self.side_group.setExclusive(True)
        self.side_group.addButton(self.btn_short); self.side_group.addButton(self.btn_long)
        modes = QtWidgets.QHBoxLayout(); modes.addWidget(self.mode_pending); modes.addWidget(self.mode_market)
        sides = QtWidgets.QHBoxLayout(); sides.addWidget(self.btn_short); sides.addWidget(self.btn_long)
        outer.addLayout(modes); outer.addLayout(sides)

        self.status_label = QtWidgets.QLabel("Choose mode + direction")
        outer.addWidget(self.status_label)
        self.btn_place = QtWidgets.QPushButton("SEND")
        self.btn_cancel_plan = QtWidgets.QPushButton("CANCEL")
        outer.addWidget(self.btn_place); outer.addWidget(self.btn_cancel_plan)
        self.btn_place.setToolTip("送單接線將在下一階段加入；本階段只預覽下單參數。")

        # Keep the existing order/replay engine and its record widgets alive while
        # the old tables are removed from the visible Order Panel.
        self.pending_table = QtWidgets.QTableWidget(0, 6)
        self.records_table = QtWidgets.QTableWidget(0, 8)
        self.lbl_unreal = QtWidgets.QLabel()
        self.lbl_real = QtWidgets.QLabel()
        self.lbl_pos = QtWidgets.QLabel()
        self.btn_close = QtWidgets.QPushButton()
        self.btn_cancel = QtWidgets.QPushButton()
        self.btn_delete_record = QtWidgets.QPushButton()
        self.btn_export = QtWidgets.QPushButton()
        self.btn_clean = QtWidgets.QPushButton()

        self.btn_place.clicked.connect(self._send_not_ready)
        self.btn_cancel_plan.clicked.connect(self.cancel_plan)
        self.risk_mode_button.clicked.connect(self.toggle_risk_mode)
        self.mode_pending.clicked.connect(lambda: self.set_order_mode("pending"))
        self.mode_market.clicked.connect(lambda: self.set_order_mode("market"))
        self.btn_long.clicked.connect(lambda: self.set_side("long"))
        self.btn_short.clicked.connect(lambda: self.set_side("short"))
        for widget in (self.equity_spin, self.risk_input):
            widget.valueChanged.connect(self.update_metrics)
        for widget in (self.entry_edit, self.sl_edit, self.tp_edit):
            widget.textChanged.connect(self.update_metrics)
        self.set_order_mode("pending")
        self.set_side("long")
        self._update_selection_styles()
        self.update_metrics()

    @property
    def selected_side(self) -> str:
        return self._selected_side

    def set_side(self, side: str) -> None:
        self._selected_side = "short" if side == "short" else "long"
        self.btn_long.setChecked(self._selected_side == "long")
        self.btn_short.setChecked(self._selected_side == "short")
        self._update_selection_styles()
        self.update_metrics()
        self._update_status()

    def set_order_mode(self, mode: str) -> None:
        market = mode == "market"
        self.mode_market.setChecked(market)
        self.mode_pending.setChecked(not market)
        self.entry_edit.setEnabled(not market)
        self._update_selection_styles()
        self.update_metrics()
        self._update_status()

    def _update_selection_styles(self) -> None:
        pending = "#2962ff"
        market = "#f5a623"
        short = "#ef5350"
        long = "#26a69a"
        for button, selected, color in (
            (self.mode_pending, self.mode_pending.isChecked(), pending),
            (self.mode_market, self.mode_market.isChecked(), market),
            (self.btn_short, self.btn_short.isChecked(), short),
            (self.btn_long, self.btn_long.isChecked(), long),
        ):
            bg = color if selected else "#202836"
            border = color if selected else "#343e50"
            button.setStyleSheet(f"QPushButton {{ background:{bg}; color:white; border:1px solid {border}; padding:6px; }}")
        side_color = short if self.selected_side == "short" else long
        self.btn_place.setStyleSheet(
            f"QPushButton {{ background:{side_color}; color:white; border:1px solid {side_color}; padding:8px; font-weight:700; }}"
            "QPushButton:disabled { color:#a0a7b2; background:#343b46; border:1px solid #414957; }"
        )

    def toggle_risk_mode(self) -> None:
        self._risk_values[self._risk_mode] = float(self.risk_input.value())
        self._risk_mode = "percent" if self._risk_mode == "cash" else "cash"
        self.risk_input.setValue(self._risk_values[self._risk_mode])
        self.risk_mode_button.setText("Risk %" if self._risk_mode == "percent" else "Risk $")
        self.update_metrics()

    def _update_status(self) -> None:
        mode = "PENDING" if self.mode_pending.isChecked() else "MARKET"
        side = self.selected_side.upper()
        color = "#ef5350" if side == "SHORT" else "#26a69a"
        self.status_label.setText(f"{mode}  •  {side}")
        self.status_label.setStyleSheet(f"color:{color}; font-weight:700;")

    def _send_not_ready(self) -> None:
        self.status_label.setText("Preview only — order submission will be connected in the next step")
        self.status_label.setStyleSheet("color:#f5a623;")

    def cancel_plan(self) -> None:
        self.entry_edit.setText("0")
        self.sl_edit.setText("0")
        self.tp_edit.setText("0")
        self.status_label.setText("Plan cancelled")
        self.status_label.setStyleSheet("color:#9aa9bf;")
        self.update_metrics()

    @staticmethod
    def _read_price(text: str) -> float | None:
        try:
            value = float(text.strip().replace(",", ""))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _entry_price(self) -> float | None:
        if self.mode_market.isChecked():
            value = self.current_price()
            return None if value is None else float(value)
        return self._read_price(self.entry_edit.text())

    def _risk_target(self) -> float:
        value = float(self.risk_input.value())
        if self._risk_mode == "percent":
            return float(self.equity_spin.value()) * value / 100.0
        return value

    def _order_type(self, entry: float | None) -> str:
        if self.mode_market.isChecked():
            return "market"
        current = self.current_price()
        if entry is None or current is None:
            return "--"
        if self.selected_side == "long":
            return "limit" if entry <= current else "stop market"
        return "limit" if entry >= current else "stop market"

    def update_metrics(self, *_args) -> None:
        entry = self._entry_price()
        if self.mode_market.isChecked() and entry is not None:
            blocked = self.entry_edit.blockSignals(True)
            self.entry_edit.setText(f"{entry:.{self._price_precision(entry)}f}")
            self.entry_edit.blockSignals(blocked)
        sl = self._read_price(self.sl_edit.text())
        tp = self._read_price(self.tp_edit.text())
        risk_target = self._risk_target()
        entry_edit = entry if entry is not None else 0.0
        self.price_edit.setText(str(entry_edit))
        order_type = self._order_type(entry)
        self.order_type_label.setText(f"Order Type: {order_type}")
        if order_type in ("market", "limit", "stop market"):
            self.type_combo.setCurrentText(order_type)

        multiplier = _side_mult(self.selected_side)
        stop_valid = sl is not None and entry is not None and (sl < entry if multiplier > 0 else sl > entry)
        lots = None
        est_loss = None
        est_profit = None
        rr = None
        if risk_target > 0 and stop_valid:
            per_lot_loss = abs(entry - sl)
            if per_lot_loss > 0:
                lots = risk_target / per_lot_loss
                est_loss = lots * per_lot_loss
                if tp is not None and (tp > entry if multiplier > 0 else tp < entry):
                    est_profit = abs(tp - entry) * lots
                    rr = abs(tp - entry) / per_lot_loss

        self.qty_spin.setValue(0.0 if lots is None else lots)
        self.metric_labels["Est Loss"].setText("--" if est_loss is None else f"{est_loss:,.2f} USD")
        self.metric_labels["Est Profit"].setText("--" if est_profit is None else f"{est_profit:,.2f} USD")
        self.metric_labels["RR"].setText("--" if rr is None else f"{rr:.2f}")
        self.metric_labels["Risk Target"].setText(f"{risk_target:,.2f} USD" if risk_target > 0 else "--")
        self.metric_labels["Lots"].setText("--" if lots is None else f"{lots:.6f}")

    def set_context(self, case, raw_df: pd.DataFrame, replay):
        self.case = case
        self.raw_df = raw_df
        self.replay = replay
        # Unfilled orders are not research records and are never restored.
        self.pending_orders.clear()
        # Compatibility cleanup for v0.1.3 cases that may contain open/cancelled rows.
        if self.case is not None:
            filled_only = [o for o in self.case.orders if o.get("status") == "filled"]
            if len(filled_only) != len(self.case.orders):
                self.case.orders = filled_only
                self.case.touch()
                self.changed.emit()
        self.refresh()

    def _current_ts(self) -> float:
        return float(self.replay.current_ts) if self.replay is not None else 0.0

    def _current_time_text(self) -> str:
        if self.replay is None:
            return ""
        tz = self.case.display.get("timezone", "Asia/Taipei") if self.case is not None else "UTC"
        try:
            return pd.Timestamp(self.replay.current_ts, unit="s", tz="UTC").tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            return pd.Timestamp(self.replay.current_ts, unit="s", tz="UTC").isoformat()

    def current_price(self):
        if self.replay is None or self.raw_df.empty:
            return None
        sub = self.raw_df[self.raw_df["timestamp"] <= self.replay.current_ts]
        if sub.empty:
            return None
        return float(sub.iloc[-1]["close"])

    def _price_precision(self, price: float | None = None) -> int:
        value = self.current_price() if price is None else price
        if value is None:
            return 2
        return 4 if abs(float(value)) < 10 else 2

    def _new_order_record(self, side: str, order_type: str, requested_price, qty: float, origin="place"):
        ts = self._current_ts()
        return {
            "id": f"order-{uuid.uuid4().hex[:12]}",
            "created_at": _utc_now_iso(),
            "created_ts": ts,
            "replay_time": self._current_time_text(),
            "side": side,
            "order_type": order_type,
            "requested_price": None if requested_price is None else float(requested_price),
            "qty": float(qty),
            "origin": origin,
            "status": "open",
            "fill_ts": None,
            "fill_time": None,
            "fill_price": None,
            "cancel_ts": None,
            "cancel_time": None,
            "computed_action": "",
            "realized_pnl": 0.0,
            "realized_r_pnl": 0.0,
        }

    def place_order(self):
        if self.case is None or self.replay is None:
            return
        side = self.selected_side
        order_type = self.type_combo.currentText()
        qty = float(self.qty_spin.value())
        if qty <= 0:
            QtWidgets.QMessageBox.warning(self, "Order", "Quantity 必須大於 0")
            return
        cp = self.current_price()
        if cp is None:
            QtWidgets.QMessageBox.warning(self, "Order", "目前沒有可成交價格")
            return

        requested_price = None
        if order_type != "market":
            try:
                requested_price = float(self.price_edit.text().strip())
            except Exception:
                QtWidgets.QMessageBox.warning(self, "Order", "請輸入有效價格")
                return

        record = self._new_order_record(side, order_type, requested_price, qty)
        if order_type == "market":
            # Market orders are persisted only after the fill exists.
            self._fill_record(record, cp, self._current_ts())
            self.case.orders.append(record)
            self.case.touch()
            self.refresh()
            self.changed.emit(); self.fills_changed.emit()
            return

        # Limit / stop-market orders remain in memory until they actually fill.
        # A cancelled or never-filled order therefore never enters Case JSON.
        self.pending_orders.append(record)
        self.refresh()

    def close_position(self):
        if self.case is None or self.replay is None:
            return
        self._recompute_state()
        if self._position is None:
            return
        cp = self.current_price()
        if cp is None:
            return
        side = "short" if self._position["side"] == "long" else "long"
        qty = float(self._position["qty"])
        record = self._new_order_record(side, "market", None, qty, origin="manual-close")
        self._fill_record(record, cp, self._current_ts())
        self.case.orders.append(record)
        self.case.touch()
        self.refresh()
        self.changed.emit(); self.fills_changed.emit()

    def _fill_record(self, record: dict, price: float, fill_ts: float):
        record["status"] = "filled"
        record["fill_price"] = float(price)
        record["fill_ts"] = float(fill_ts)
        tz = self.case.display.get("timezone", "Asia/Taipei") if self.case is not None else "UTC"
        try:
            record["fill_time"] = pd.Timestamp(fill_ts, unit="s", tz="UTC").tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            record["fill_time"] = pd.Timestamp(fill_ts, unit="s", tz="UTC").isoformat()

    def process_replay_advance(self, previous_ts: float, current_ts: float):
        """Fill legacy limit/stop orders when a newly revealed M1 bar touches the order price."""
        if self.case is None or self.raw_df.empty or current_ts <= previous_ts:
            self.refresh()
            return
        bars = self.raw_df[(self.raw_df["timestamp"] > previous_ts) & (self.raw_df["timestamp"] <= current_ts)]
        changed = False
        for _, bar in bars.iterrows():
            bar_ts = float(bar["timestamp"])
            low = float(bar["low"]); high = float(bar["high"])
            for record in list(self.pending_orders):
                if float(record.get("created_ts") or 0.0) > bar_ts:
                    continue
                if record.get("order_type") not in ("limit", "stop market"):
                    continue
                price = record.get("requested_price")
                if price is None:
                    continue
                price = float(price)
                if low <= price <= high:
                    self._fill_record(record, price, bar_ts)
                    self.pending_orders.remove(record)
                    self.case.orders.append(record)
                    changed = True
        if changed:
            self.case.touch()
            self.changed.emit(); self.fills_changed.emit()
        self.refresh()

    def cancel_selected(self):
        if self.case is None:
            return
        row = self.pending_table.currentRow()
        if row < 0:
            return
        item = self.pending_table.item(row, 0)
        if item is None:
            return
        order_id = item.data(QtCore.Qt.UserRole) or item.text()
        for record in list(self.pending_orders):
            if record.get("id") == order_id:
                self.pending_orders.remove(record)
                self.refresh()
                return

    def delete_selected_record(self):
        if self.case is None:
            return
        row = self.records_table.currentRow()
        if row < 0:
            return
        item = self.records_table.item(row, 0)
        if item is None:
            return
        order_id = item.data(QtCore.Qt.UserRole)
        if not order_id:
            return
        self.case.orders = [o for o in self.case.orders if o.get("id") != order_id]
        self.case.touch()
        self.refresh()
        self.changed.emit(); self.fills_changed.emit()

    def export_records(self):
        if self.case is None or not self.case.orders:
            QtWidgets.QMessageBox.information(self, "Export", "No order records to export")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Order Records", "order_records.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            pd.DataFrame(self.case.orders).to_csv(path, index=False, encoding="utf-8-sig")
            QtWidgets.QMessageBox.information(self, "Export", f"Order records exported to:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(exc))

    def clean_records(self):
        if self.case is None or not self.case.orders:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Clean Records", "刪除目前 Case 的所有 Order 紀錄？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.case.orders.clear()
        self.case.touch()
        self.refresh()
        self.changed.emit(); self.fills_changed.emit()

    def _fill_qty_ratio(self, ratio: float):
        self._recompute_state()
        base = float(self._position["qty"]) if self._position is not None else 0.0
        self.qty_spin.setValue(base * ratio)

    def _recompute_state(self):
        if self.case is None:
            self._position = None; self._realized_pnl = 0.0; self._realized_r = 0.0
            return
        current_ts = self._current_ts()
        position = None
        realized = 0.0
        records = [o for o in self.case.orders if o.get("status") == "filled" and o.get("fill_ts") is not None and float(o.get("fill_ts")) <= current_ts]
        records.sort(key=lambda o: (float(o.get("fill_ts") or 0.0), float(o.get("created_ts") or 0.0), str(o.get("id", ""))))

        for order in self.case.orders:
            order["computed_action"] = ""
            order["realized_pnl"] = 0.0
            order["realized_r_pnl"] = 0.0

        for order in records:
            side = str(order.get("side"))
            price = float(order.get("fill_price"))
            qty = float(order.get("qty") or 0.0)
            if qty <= 0:
                continue
            action_parts = []
            order_realized = 0.0
            if position is None:
                position = {"side": side, "entry": price, "qty": qty}
                action_parts.append("OPEN")
            elif position["side"] == side:
                new_qty = position["qty"] + qty
                position["entry"] = (position["entry"] * position["qty"] + price * qty) / new_qty
                position["qty"] = new_qty
                action_parts.append("ADD")
            else:
                close_qty = min(position["qty"], qty)
                pnl = (price - position["entry"]) * _side_mult(position["side"]) * close_qty
                realized += pnl; order_realized += pnl
                position["qty"] -= close_qty
                action_parts.append("CLOSE")
                remaining = qty - close_qty
                if position["qty"] <= 1e-12:
                    position = None
                if remaining > 1e-12:
                    position = {"side": side, "entry": price, "qty": remaining}
                    action_parts.append("OPEN")
            order["computed_action"] = "+".join(action_parts)
            order["realized_pnl"] = float(order_realized)
            order["realized_r_pnl"] = float(order_realized / self.R_VALUE)

        self._position = position
        self._realized_pnl = realized
        self._realized_r = realized / self.R_VALUE

    def refresh(self):
        self._recompute_state()
        cp = self.current_price()
        self.update_metrics()
        unreal = 0.0
        if self._position is not None and cp is not None:
            unreal = (cp - self._position["entry"]) * _side_mult(self._position["side"]) * self._position["qty"]
        self.lbl_unreal.setText(f"Unrealized PnL: {unreal:.3f} ({unreal / self.R_VALUE:.3f}R)")
        self.lbl_real.setText(f"Realized PnL: {self._realized_pnl:.3f} ({self._realized_r:.3f}R)")
        if self._position is None:
            self.lbl_pos.setText("Position: flat")
        else:
            p = self._price_precision(self._position["entry"])
            self.lbl_pos.setText(f"Position: {self._position['side']} {self._position['qty']:.4f} @ {self._position['entry']:.{p}f}")
        self._refresh_pending_table()
        self._refresh_records_table()

    def _refresh_pending_table(self):
        pending = list(self.pending_orders)
        self.pending_table.setRowCount(len(pending))
        for r, o in enumerate(pending):
            vals = [
                str(o.get("id", ""))[-6:], o.get("side", ""), o.get("order_type", ""),
                "-" if o.get("requested_price") is None else f"{float(o['requested_price']):.{self._price_precision(o.get('requested_price'))}f}",
                f"{float(o.get('qty') or 0):.4f}", o.get("status", ""),
            ]
            for c, value in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(str(value))
                if c == 0:
                    item.setData(QtCore.Qt.UserRole, o.get("id"))
                self.pending_table.setItem(r, c, item)
        self.pending_table.resizeColumnsToContents()

    def _refresh_records_table(self):
        orders = list(self.case.orders if self.case is not None else [])
        self.records_table.setRowCount(len(orders))
        for r, o in enumerate(orders):
            display_price = o.get("fill_price") if o.get("fill_price") is not None else o.get("requested_price")
            price_text = "-" if display_price is None else f"{float(display_price):.{self._price_precision(display_price)}f}"
            pnl = float(o.get("realized_pnl") or 0.0)
            vals = [
                o.get("replay_time", ""), o.get("side", ""), o.get("order_type", ""), price_text,
                f"{float(o.get('qty') or 0):.4f}", o.get("status", ""), o.get("computed_action", ""),
                "-" if abs(pnl) < 1e-12 else f"{pnl:.3f}",
            ]
            for c, value in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(str(value))
                if c == 0:
                    item.setData(QtCore.Qt.UserRole, o.get("id"))
                self.records_table.setItem(r, c, item)
        self.records_table.resizeColumnsToContents()

    def visible_fill_events(self) -> list[dict]:
        """Filled order points at or before current replay time, for chart markers."""
        if self.case is None:
            return []
        events, _ = build_order_overlay(self.case.orders, self._current_ts())
        return events

    def visible_trade_segments(self) -> list[dict]:
        """Completed Open-to-Close segments at or before current replay time."""
        if self.case is None:
            return []
        _, segments = build_order_overlay(self.case.orders, self._current_ts())
        return segments
