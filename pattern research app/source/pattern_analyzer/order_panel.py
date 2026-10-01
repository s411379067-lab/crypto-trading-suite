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

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(5)

        title = QtWidgets.QLabel("Order")
        title.setStyleSheet("font-size:13pt; font-weight:700; color:#e6edf7;")
        outer.addWidget(title)

        form = QtWidgets.QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(4)

        self.side_combo = QtWidgets.QComboBox(); self.side_combo.addItems(["long", "short"])
        self.type_combo = QtWidgets.QComboBox(); self.type_combo.addItems(["market", "limit", "stop market"])
        self.price_edit = QtWidgets.QLineEdit(); self.price_edit.setPlaceholderText("Price")
        self.qty_spin = QtWidgets.QDoubleSpinBox()
        self.qty_spin.setDecimals(4); self.qty_spin.setRange(0.0, 1_000_000.0); self.qty_spin.setValue(1.0)

        form.addWidget(QtWidgets.QLabel("Side"), 0, 0); form.addWidget(self.side_combo, 0, 1)
        form.addWidget(QtWidgets.QLabel("Type"), 1, 0); form.addWidget(self.type_combo, 1, 1)
        form.addWidget(QtWidgets.QLabel("Price"), 2, 0); form.addWidget(self.price_edit, 2, 1)
        form.addWidget(QtWidgets.QLabel("Quantity"), 3, 0); form.addWidget(self.qty_spin, 3, 1)
        outer.addLayout(form)

        ratio_row = QtWidgets.QHBoxLayout()
        self.btn_1_3 = QtWidgets.QPushButton("1/3 P")
        self.btn_1_2 = QtWidgets.QPushButton("1/2 P")
        self.btn_full = QtWidgets.QPushButton("full P")
        ratio_row.addWidget(self.btn_1_3); ratio_row.addWidget(self.btn_1_2); ratio_row.addWidget(self.btn_full)
        outer.addLayout(ratio_row)

        action_row = QtWidgets.QHBoxLayout()
        self.btn_place = QtWidgets.QPushButton("Place Order")
        self.btn_close = QtWidgets.QPushButton("Close Position")
        action_row.addWidget(self.btn_place); action_row.addWidget(self.btn_close)
        outer.addLayout(action_row)

        self.lbl_unreal = QtWidgets.QLabel("Unrealized PnL: 0.000 (0.000R)")
        self.lbl_real = QtWidgets.QLabel("Realized PnL: 0.000 (0.000R)")
        self.lbl_pos = QtWidgets.QLabel("Position: flat")
        outer.addWidget(self.lbl_unreal); outer.addWidget(self.lbl_real); outer.addWidget(self.lbl_pos)

        pending_box = QtWidgets.QGroupBox("Pending Orders")
        pending_layout = QtWidgets.QVBoxLayout(pending_box)
        self.pending_table = QtWidgets.QTableWidget(0, 6)
        self.pending_table.setHorizontalHeaderLabels(["ID", "Side", "Type", "Price", "Qty", "Status"])
        self.pending_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.pending_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.pending_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.pending_table.horizontalHeader().setStretchLastSection(True)
        self._style_order_table(self.pending_table)
        self.pending_table.setMaximumHeight(135)
        self.btn_cancel = QtWidgets.QPushButton("Cancel Selected")
        pending_layout.addWidget(self.pending_table); pending_layout.addWidget(self.btn_cancel)
        outer.addWidget(pending_box)

        records_box = QtWidgets.QGroupBox("Order Records")
        records_layout = QtWidgets.QVBoxLayout(records_box)
        self.records_table = QtWidgets.QTableWidget(0, 8)
        self.records_table.setHorizontalHeaderLabels(["Time", "Side", "Type", "Price", "Qty", "Status", "Action", "PnL"])
        self.records_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.records_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.records_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.records_table.horizontalHeader().setStretchLastSection(True)
        self._style_order_table(self.records_table)
        self.records_table.setMinimumHeight(145)
        self.btn_delete_record = QtWidgets.QPushButton("Delete Selected Record")
        records_layout.addWidget(self.records_table, 1); records_layout.addWidget(self.btn_delete_record)
        history_buttons = QtWidgets.QHBoxLayout()
        self.btn_export = QtWidgets.QPushButton("Export Records")
        self.btn_clean = QtWidgets.QPushButton("Clean Records")
        history_buttons.addWidget(self.btn_export); history_buttons.addWidget(self.btn_clean)
        records_layout.addLayout(history_buttons)
        outer.addWidget(records_box, 1)

        self.btn_place.clicked.connect(self.place_order)
        self.btn_close.clicked.connect(self.close_position)
        self.btn_cancel.clicked.connect(self.cancel_selected)
        self.btn_delete_record.clicked.connect(self.delete_selected_record)
        self.btn_export.clicked.connect(self.export_records)
        self.btn_clean.clicked.connect(self.clean_records)
        self.btn_1_3.clicked.connect(lambda: self._fill_qty_ratio(1.0 / 3.0))
        self.btn_1_2.clicked.connect(lambda: self._fill_qty_ratio(1.0 / 2.0))
        self.btn_full.clicked.connect(lambda: self._fill_qty_ratio(1.0))

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

    def prefill_from_chart(self, side: str, price: float):
        """Legacy S/L axis-button behavior: choose limit vs stop-market from current price."""
        cp = self.current_price()
        if cp is None:
            return
        side = "short" if side == "short" else "long"
        price = float(price)
        if cp > price:
            order_type = "stop market" if side == "short" else "limit"
        else:
            order_type = "limit" if side == "short" else "stop market"
        self.side_combo.setCurrentText(side)
        self.type_combo.setCurrentText(order_type)
        self.price_edit.setText(f"{price:.{self._price_precision(price)}f}")

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
        side = self.side_combo.currentText()
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
