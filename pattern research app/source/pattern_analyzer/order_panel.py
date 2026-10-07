from __future__ import annotations

import uuid
from copy import deepcopy
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
    order_plan_changed = QtCore.Signal(dict)

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
        self._active_entry_id: str | None = None
        self._confirmed_bracket: tuple[float, float] | None = None
        self._bracket_groups: list[dict] = []
        self._confirmed_bracket_groups: list[dict] | None = None
        self._next_bracket_group_number = 1
        self._manual_group_allocation = False
        self._group_plan_initialized = False
        self._last_plan_lots: float | None = None
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

        self.bracket_group_box = QtWidgets.QGroupBox("SL / TP Groups")
        bracket_group_layout = QtWidgets.QVBoxLayout(self.bracket_group_box)
        self.bracket_group_table = QtWidgets.QTableWidget(0, 4)
        self.bracket_group_table.setHorizontalHeaderLabels(["Group", "SL", "TP", "Lots"])
        self.bracket_group_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.bracket_group_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.bracket_group_table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.bracket_group_table.horizontalHeader().setStretchLastSection(True)
        self.bracket_group_table.verticalHeader().setVisible(False)
        self.bracket_group_table.setMaximumHeight(130)
        self._style_order_table(self.bracket_group_table)
        self.group_allocation_label = QtWidgets.QLabel("Assigned Lots: --")
        group_buttons = QtWidgets.QHBoxLayout()
        self.btn_add_bracket_group = QtWidgets.QPushButton("Add SL/TP")
        self.btn_remove_bracket_group = QtWidgets.QPushButton("Remove Group")
        group_buttons.addWidget(self.btn_add_bracket_group)
        group_buttons.addWidget(self.btn_remove_bracket_group)
        bracket_group_layout.addWidget(self.bracket_group_table)
        bracket_group_layout.addWidget(self.group_allocation_label)
        bracket_group_layout.addLayout(group_buttons)
        outer.addWidget(self.bracket_group_box)

        self.status_label = QtWidgets.QLabel("Choose mode + direction")
        outer.addWidget(self.status_label)
        self.btn_place = QtWidgets.QPushButton("SEND")
        self.btn_cancel_plan = QtWidgets.QPushButton("CANCEL")
        outer.addWidget(self.btn_place); outer.addWidget(self.btn_cancel_plan)
        self.btn_place.setToolTip("送出目前的下單計畫")

        self.pending_table = QtWidgets.QTableWidget(0, 6)
        self.pending_table.setHorizontalHeaderLabels(["ID", "Side", "Type", "Price", "Qty", "Status"])
        self.pending_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.pending_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.pending_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.pending_table.horizontalHeader().setStretchLastSection(True)
        self._style_order_table(self.pending_table)
        self.pending_table.setMinimumHeight(70)
        self.pending_box = QtWidgets.QGroupBox("Pending Orders")
        pending_layout = QtWidgets.QVBoxLayout(self.pending_box)
        self.btn_cancel = QtWidgets.QPushButton("Cancel Selected")
        pending_layout.addWidget(self.pending_table)
        pending_layout.addWidget(self.btn_cancel)

        self.records_table = QtWidgets.QTableWidget(0, 8)
        self.records_table.setHorizontalHeaderLabels(["Time", "Side", "Type", "Price", "Qty", "Status", "Action", "PnL"])
        self.records_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.records_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.records_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.records_table.horizontalHeader().setStretchLastSection(True)
        self._style_order_table(self.records_table)
        self.records_table.setMinimumHeight(80)
        self.records_box = QtWidgets.QGroupBox("Order Records")
        records_layout = QtWidgets.QVBoxLayout(self.records_box)
        self.btn_delete_record = QtWidgets.QPushButton("Delete Selected Record")
        records_layout.addWidget(self.records_table)
        records_layout.addWidget(self.btn_delete_record)
        history_buttons = QtWidgets.QHBoxLayout()
        self.btn_export = QtWidgets.QPushButton("Export Records")
        self.btn_clean = QtWidgets.QPushButton("Clean Records")
        history_buttons.addWidget(self.btn_export)
        history_buttons.addWidget(self.btn_clean)
        records_layout.addLayout(history_buttons)
        self.history_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        self.history_splitter.setHandleWidth(7)
        self.history_splitter.setChildrenCollapsible(False)
        self.history_splitter.addWidget(self.pending_box)
        self.history_splitter.addWidget(self.records_box)
        self.history_splitter.setStretchFactor(0, 1)
        self.history_splitter.setStretchFactor(1, 2)
        self.history_splitter.setSizes([145, 220])
        outer.addWidget(self.history_splitter, 1)

        self.lbl_unreal = QtWidgets.QLabel()
        self.lbl_real = QtWidgets.QLabel()
        self.lbl_pos = QtWidgets.QLabel()
        self.btn_close = QtWidgets.QPushButton()

        self.btn_place.clicked.connect(self.submit_plan)
        self.btn_cancel_plan.clicked.connect(self.cancel_plan)
        self.btn_add_bracket_group.clicked.connect(self.add_bracket_group)
        self.btn_remove_bracket_group.clicked.connect(self.remove_selected_bracket_group)
        self.bracket_group_table.cellChanged.connect(self._bracket_group_cell_changed)
        self.btn_cancel.clicked.connect(self.cancel_selected)
        self.btn_delete_record.clicked.connect(self.delete_selected_record)
        self.btn_export.clicked.connect(self.export_records)
        self.btn_clean.clicked.connect(self.clean_records)
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

    def _set_send_status(self, message: str, color: str) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color:{color}; font-weight:700;")

    def submit_plan(self) -> bool:
        """Submit the panel plan through the existing simulated-order lifecycle."""
        if self.case is None or self.replay is None:
            self._set_send_status("Open a Case and start replay before sending", "#f5a623")
            return False
        self._recompute_state()
        if self._active_entry_id is not None or self._position is not None:
            self._set_send_status("Manage or close the current position before sending another", "#f5a623")
            return False

        entry = self._entry_price()
        sl = self._read_price(self.sl_edit.text())
        tp = self._read_price(self.tp_edit.text())
        qty = float(self.qty_spin.value())
        if entry is None:
            self._set_send_status("Enter a valid Entry price", "#ef5350")
            return False
        if sl is None or tp is None:
            self._set_send_status("Enter valid SL and TP prices", "#ef5350")
            return False
        is_long = self.selected_side == "long"
        if (sl >= entry if is_long else sl <= entry):
            self._set_send_status("SL must be below Entry for Long and above Entry for Short", "#ef5350")
            return False
        if (tp <= entry if is_long else tp >= entry):
            self._set_send_status("TP must be above Entry for Long and below Entry for Short", "#ef5350")
            return False
        if qty <= 0:
            self._set_send_status("Lots must be greater than 0 (check Risk and SL)", "#ef5350")
            return False
        if not self._bracket_groups:
            self._bracket_groups = [{
                "id": "group-1", "label": "Group 1", "sl": float(sl), "tp": float(tp), "qty": qty,
            }]
            self._next_bracket_group_number = max(self._next_bracket_group_number, 2)
        groups = deepcopy(self._bracket_groups)
        assigned_lots = sum(float(group.get("qty") or 0.0) for group in groups)
        if assigned_lots <= 1e-12:
            self._set_send_status("Allocate Lots to at least one SL/TP group", "#ef5350")
            return False
        if assigned_lots > qty + 1e-9:
            self._set_send_status("Group Lots total cannot exceed position Lots", "#ef5350")
            return False
        for group in groups:
            group_sl = float(group.get("sl") or 0.0)
            group_tp = float(group.get("tp") or 0.0)
            if group_sl <= 0 or group_tp <= 0:
                self._set_send_status("Every SL/TP group needs valid prices", "#ef5350")
                return False
            if (group_sl >= entry if is_long else group_sl <= entry):
                self._set_send_status("A group SL is on the wrong side of Entry", "#ef5350")
                return False
            if (group_tp <= entry if is_long else group_tp >= entry):
                self._set_send_status("A group TP is on the wrong side of Entry", "#ef5350")
                return False

        current_price = self.current_price()
        if current_price is None:
            self._set_send_status("No replay price is available", "#ef5350")
            return False

        market = self.mode_market.isChecked()
        order_type = "market" if market else self._order_type(entry)
        if order_type not in ("market", "limit", "stop market"):
            self._set_send_status("Could not determine pending order type", "#ef5350")
            return False

        record = self._new_order_record(
            self.selected_side,
            order_type,
            None if market else entry,
            qty,
        )
        record["bracket_groups"] = groups
        record["stop_loss"] = float(groups[0]["sl"])
        record["take_profit"] = float(groups[0]["tp"])
        record["est_loss"] = float(self._risk_target())
        record["bracket_status"] = "waiting-entry" if not market else "active"
        self._active_entry_id = record["id"]
        self._confirmed_bracket = (float(sl), float(tp))
        self._confirmed_bracket_groups = deepcopy(groups)
        self._bracket_groups = deepcopy(groups)
        if market:
            self._fill_record(record, current_price, self._current_ts())
            self.case.orders.append(record)
            self._create_protection_orders(record, active=True, active_ts=self._current_ts())
            self.case.touch()
            self.refresh()
            self.changed.emit()
            self.fills_changed.emit()
            self._set_send_status(f"Market {self.selected_side.upper()} filled at {current_price:g}", "#7bd88f")
            return True

        self.pending_orders.append(record)
        self._create_protection_orders(record, active=False)
        self.refresh()
        self._set_send_status(f"{self.selected_side.upper()} {order_type.upper()} order sent", "#7bd88f")
        return True

    def _create_protection_orders(self, entry: dict, *, active: bool, active_ts: float | None = None) -> None:
        """Create SL/TP rows tied to one entry; pending-entry brackets stay unarmed."""
        close_side = "short" if entry["side"] == "long" else "long"
        entry["bracket_order_ids"] = []
        groups = entry.get("bracket_groups") if "bracket_groups" in entry else [{
            "id": "group-1", "label": "Group 1", "sl": entry.get("stop_loss"),
            "tp": entry.get("take_profit"), "qty": entry.get("qty", 0.0),
        }]
        for group in groups:
            qty = float(group.get("qty") or 0.0)
            if qty <= 1e-12:
                continue
            for role, order_type, key in (
                ("stop_loss", "stop market", "sl"),
                ("take_profit", "limit", "tp"),
            ):
                order = self._new_order_record(close_side, order_type, group[key], qty, origin=f"bracket-{role}")
                order.update({
                    "role": role,
                    "group_id": group["id"],
                    "parent_order_id": entry["id"],
                    "status": "open",
                    "armed": bool(active),
                    "active_from_ts": active_ts,
                })
                entry["bracket_order_ids"].append(order["id"])
                self.pending_orders.append(order)

    def _refresh_bracket_group_table(self, total_lots: float | None = None) -> None:
        table = self.bracket_group_table
        blocked = table.blockSignals(True)
        table.setRowCount(len(self._bracket_groups))
        assigned = sum(float(group.get("qty") or 0.0) for group in self._bracket_groups)
        active_entry = self._active_entry_record()
        locked = bool(active_entry and active_entry.get("status") != "filled")
        for row, group in enumerate(self._bracket_groups):
            values = [
                str(group.get("label") or group.get("id", "Group")),
                f"{float(group.get('sl') or 0.0):,.2f}",
                f"{float(group.get('tp') or 0.0):,.2f}",
                f"{float(group.get('qty') or 0.0):.6f}",
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col != 3:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                else:
                    item.setToolTip("Double-click to edit the Lots allocated to this SL/TP pair")
                    if locked:
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                if col == 0:
                    item.setData(QtCore.Qt.UserRole, group.get("id"))
                table.setItem(row, col, item)
        table.blockSignals(blocked)
        if total_lots is None:
            self.group_allocation_label.setText(f"Assigned Lots: {assigned:.6f}")
        else:
            difference = float(total_lots) - assigned
            if difference >= -1e-9:
                self.group_allocation_label.setText(
                    f"Assigned: {assigned:.6f} / Position: {float(total_lots):.6f}  •  Unprotected: {max(0.0, difference):.6f}"
                )
            else:
                self.group_allocation_label.setText(
                    f"Assigned: {assigned:.6f} / Position: {float(total_lots):.6f}  •  Over by: {-difference:.6f}"
                )

    @staticmethod
    def _bracket_group_signature(groups: list[dict] | None) -> tuple:
        return tuple(
            (str(group.get("id")), round(float(group.get("sl") or 0.0), 10),
             round(float(group.get("tp") or 0.0), 10), round(float(group.get("qty") or 0.0), 8))
            for group in (groups or [])
        )

    def _bracket_group_cell_changed(self, row: int, column: int) -> None:
        if column != 3 or not 0 <= row < len(self._bracket_groups):
            return
        try:
            qty = float(self.bracket_group_table.item(row, column).text().replace(",", ""))
        except (AttributeError, TypeError, ValueError):
            self._refresh_bracket_group_table(float(self.qty_spin.value()))
            return
        if qty < 0:
            self._refresh_bracket_group_table(float(self.qty_spin.value()))
            return
        group = self._bracket_groups[row]
        previous = float(group.get("qty") or 0.0)
        group["qty"] = qty
        total = sum(float(item.get("qty") or 0.0) for item in self._bracket_groups)
        if total > float(self.qty_spin.value()) + 1e-9:
            group["qty"] = previous
            self._set_send_status("Group Lots total cannot exceed position Lots", "#ef5350")
        else:
            self._manual_group_allocation = True
            self.update_metrics()
        self._refresh_bracket_group_table(float(self.qty_spin.value()))

    def add_bracket_group(self) -> bool:
        if self._active_entry_id is not None and not self._active_bracket_is_live():
            return False
        sl = self._read_price(self.sl_edit.text())
        tp = self._read_price(self.tp_edit.text())
        if sl is None or tp is None:
            self._set_send_status("Set valid SL and TP prices before adding a group", "#f5a623")
            return False
        if not self._bracket_groups:
            total = float(self.qty_spin.value())
            self._bracket_groups.append({
                "id": f"group-{self._next_bracket_group_number}",
                "label": f"Group {self._next_bracket_group_number}",
                "sl": sl, "tp": tp, "qty": total,
            })
            self._next_bracket_group_number += 1
        else:
            largest = max(self._bracket_groups, key=lambda item: float(item.get("qty") or 0.0))
            half = float(largest.get("qty") or 0.0) / 2.0
            largest["qty"] = half
            number = self._next_bracket_group_number
            self._bracket_groups.append({
                "id": f"group-{number}", "label": f"Group {number}",
                "sl": float(largest.get("sl") or sl), "tp": float(largest.get("tp") or tp), "qty": half,
            })
            self._next_bracket_group_number += 1
        self._group_plan_initialized = True
        self._manual_group_allocation = True
        self.update_metrics()
        return True

    def remove_selected_bracket_group(self) -> bool:
        if self._active_entry_id is not None and not self._active_bracket_is_live():
            return False
        row = self.bracket_group_table.currentRow()
        if row < 0 and self._bracket_groups:
            row = len(self._bracket_groups) - 1
        if not 0 <= row < len(self._bracket_groups):
            return False
        self._bracket_groups.pop(row)
        self._group_plan_initialized = True
        self._manual_group_allocation = True
        self.update_metrics()
        return True

    def _active_bracket_is_live(self) -> bool:
        entry = self._active_entry_record()
        return bool(entry and entry.get("status") == "filled" and entry.get("bracket_status") == "active")

    def confirm_bracket_update(self) -> bool:
        entry = self._active_entry_record()
        if (
            entry is None
            or entry.get("status") != "filled"
            or entry.get("bracket_status") != "active"
            or self._confirmed_bracket is None
        ):
            return False
        groups = deepcopy(self._bracket_groups)
        if self._bracket_group_signature(groups) == self._bracket_group_signature(self._confirmed_bracket_groups):
            return False
        entry_price = float(entry.get("fill_price") or entry.get("requested_price") or 0.0)
        is_long = entry.get("side") == "long"
        self._recompute_state()
        total_qty = float(self._position.get("qty") or 0.0) if self._position is not None else 0.0
        assigned = sum(float(group.get("qty") or 0.0) for group in groups)
        if assigned > total_qty + 1e-9:
            self._set_send_status("Group Lots total cannot exceed position Lots", "#ef5350")
            return False
        for group in groups:
            sl = float(group.get("sl") or 0.0)
            tp = float(group.get("tp") or 0.0)
            if float(group.get("qty") or 0.0) <= 0:
                continue
            if sl <= 0 or tp <= 0 or (sl >= entry_price if is_long else sl <= entry_price):
                self._set_send_status("A group SL is on the wrong side of Entry", "#ef5350")
                return False
            if tp <= 0 or (tp <= entry_price if is_long else tp >= entry_price):
                self._set_send_status("A group TP is on the wrong side of Entry", "#ef5350")
                return False

        entry["bracket_groups"] = groups
        if groups:
            entry["stop_loss"] = float(groups[0]["sl"])
            entry["take_profit"] = float(groups[0]["tp"])
        self._remove_bracket_orders(entry["id"])
        self._create_protection_orders(entry, active=True, active_ts=float(entry.get("fill_ts") or self._current_ts()))
        entry["bracket_status"] = "active"
        self._bracket_groups = deepcopy(groups)
        self._confirmed_bracket_groups = deepcopy(groups)
        self._confirmed_bracket = (float(groups[0]["sl"]), float(groups[0]["tp"])) if groups else None
        self.case.touch()
        self.refresh()
        self.changed.emit()
        self._set_send_status("SL / TP changes confirmed", "#7bd88f")
        return True

    def cancel_bracket_update(self) -> bool:
        entry = self._active_entry_record()
        if entry is None or entry.get("status") != "filled" or entry.get("bracket_status") != "active" or self._confirmed_bracket_groups is None:
            return False
        if self._bracket_group_signature(self._bracket_groups) == self._bracket_group_signature(self._confirmed_bracket_groups):
            return False
        self._bracket_groups = deepcopy(self._confirmed_bracket_groups)
        if self._bracket_groups:
            sl, tp = float(self._bracket_groups[0]["sl"]), float(self._bracket_groups[0]["tp"])
            self.sl_edit.setText(f"{sl:.{self._price_precision(sl)}f}")
            self.tp_edit.setText(f"{tp:.{self._price_precision(tp)}f}")
        self.refresh()
        self._set_send_status("SL / TP changes cancelled", "#9aa9bf")
        return True

    def _active_entry_record(self) -> dict | None:
        if self.case is None or self._active_entry_id is None:
            return None
        for record in self.case.orders:
            if record.get("id") == self._active_entry_id:
                return record
        for record in self.pending_orders:
            if record.get("id") == self._active_entry_id:
                return record
        return None

    def _set_active_bracket(self, entry: dict, active: bool, active_ts: float | None = None) -> None:
        entry["bracket_status"] = "active" if active else "waiting-entry"
        self._active_entry_id = entry.get("id") if active else None
        self._confirmed_bracket = (
            (float(entry["stop_loss"]), float(entry["take_profit"])) if active else None
        )
        self._bracket_groups = deepcopy(entry.get("bracket_groups") or []) if active else []
        self._confirmed_bracket_groups = deepcopy(self._bracket_groups) if active else None
        self._group_plan_initialized = bool(self._bracket_groups)
        self._manual_group_allocation = bool(self._bracket_groups)
        for order in self.pending_orders:
            if order.get("parent_order_id") == entry.get("id"):
                order["armed"] = bool(active)
                order["active_from_ts"] = active_ts if active else None

    def _remove_bracket_orders(self, parent_id: str) -> None:
        self.pending_orders = [
            order for order in self.pending_orders
            if order.get("parent_order_id") != parent_id
        ]

    def _remove_bracket_group(self, parent_id: str, group_id: str) -> None:
        parent = self._find_entry_by_id(parent_id)
        self.pending_orders = [
            order for order in self.pending_orders
            if not (order.get("parent_order_id") == parent_id and order.get("group_id") == group_id)
        ]
        if parent is None:
            return
        parent["bracket_groups"] = [
            group for group in parent.get("bracket_groups", [])
            if group.get("id") != group_id
        ]
        if parent.get("bracket_groups"):
            parent["stop_loss"] = float(parent["bracket_groups"][0]["sl"])
            parent["take_profit"] = float(parent["bracket_groups"][0]["tp"])
        if parent_id == self._active_entry_id:
            self._bracket_groups = deepcopy(parent.get("bracket_groups", []))
            self._confirmed_bracket_groups = deepcopy(self._bracket_groups)
            self._confirmed_bracket = (
                (float(self._bracket_groups[0]["sl"]), float(self._bracket_groups[0]["tp"]))
                if self._bracket_groups else None
            )
            self._group_plan_initialized = True
            self._manual_group_allocation = True

    def _complete_protection_group(self, parent_id: str, group_id: str, exit_order: dict) -> None:
        parent = self._find_entry_by_id(parent_id)
        self._remove_bracket_group(parent_id, group_id)
        if parent is None:
            return
        parent.setdefault("bracket_exits", {})[group_id] = exit_order["id"]
        self._recompute_state()
        if self._position is None:
            self._finish_active_bracket(parent_id, "closed")
        else:
            parent["bracket_status"] = "active"
            self._active_entry_id = parent_id
            self._bracket_groups = deepcopy(parent.get("bracket_groups", []))
            self._confirmed_bracket_groups = deepcopy(self._bracket_groups)
            self._confirmed_bracket = (
                (float(self._bracket_groups[0]["sl"]), float(self._bracket_groups[0]["tp"]))
                if self._bracket_groups else None
            )

    def _find_entry_by_id(self, order_id: str | None) -> dict | None:
        if self.case is None or not order_id:
            return None
        for order in self.case.orders:
            if order.get("id") == order_id:
                return order
        for order in self.pending_orders:
            if order.get("id") == order_id and not order.get("role"):
                return order
        return None

    def _finish_active_bracket(self, parent_id: str, status: str) -> None:
        parent = self._find_entry_by_id(parent_id)
        if parent is not None:
            parent["bracket_status"] = status
        self._remove_bracket_orders(parent_id)
        if self._active_entry_id == parent_id:
            self._active_entry_id = None
            self._confirmed_bracket = None
            self._confirmed_bracket_groups = None
            self._bracket_groups = []
            self._group_plan_initialized = False
            self._manual_group_allocation = False

    def _restore_active_bracket(self) -> None:
        """Rebuild session-only protection rows for the currently open saved position."""
        if self._active_entry_id is not None:
            return
        self._recompute_state()
        if self._position is None or self.case is None:
            return
        for entry in reversed(self.case.orders):
            if (
                entry.get("status") == "filled"
                and entry.get("bracket_status") == "active"
                and entry.get("side") == self._position["side"]
                and float(entry.get("fill_ts") or 0.0) <= self._current_ts()
            ):
                self._active_entry_id = entry.get("id")
                self._bracket_groups = deepcopy(entry.get("bracket_groups") or [])
                self._confirmed_bracket_groups = deepcopy(self._bracket_groups)
                self._confirmed_bracket = (
                    (float(self._bracket_groups[0]["sl"]), float(self._bracket_groups[0]["tp"]))
                    if self._bracket_groups else None
                )
                self._group_plan_initialized = True
                self._manual_group_allocation = True
                self._create_protection_orders(entry, active=True, active_ts=float(entry.get("fill_ts") or 0.0))
                break

    def cancel_plan(self) -> None:
        active_entry = self._active_entry_record()
        if active_entry is not None:
            if active_entry.get("status") == "filled" and active_entry.get("bracket_status") == "active":
                self.cancel_bracket_update()
            else:
                self._set_send_status("Cancel the pending entry from Pending Orders", "#f5a623")
            return
        self.entry_edit.setText("0")
        self.sl_edit.setText("0")
        self.tp_edit.setText("0")
        self._bracket_groups = []
        self._confirmed_bracket_groups = None
        self._group_plan_initialized = False
        self._manual_group_allocation = False
        self._next_bracket_group_number = 1
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
        active_entry = self._active_entry_record()
        submitted_pending_entry = bool(active_entry and active_entry.get("status") == "open")
        plan_side = str(active_entry.get("side")) if active_entry else self.selected_side
        bracket_active = bool(
            active_entry
            and active_entry.get("status") == "filled"
            and active_entry.get("bracket_status") == "active"
        )
        if submitted_pending_entry:
            entry = active_entry.get("requested_price")
        if bracket_active:
            entry = float(active_entry.get("fill_price") or entry or 0.0)
        if self.mode_market.isChecked() and entry is not None and active_entry is None:
            blocked = self.entry_edit.blockSignals(True)
            self.entry_edit.setText(f"{entry:.{self._price_precision(entry)}f}")
            self.entry_edit.blockSignals(blocked)
        sl = self._read_price(self.sl_edit.text())
        tp = self._read_price(self.tp_edit.text())
        risk_target = self._risk_target()
        entry_edit = entry if entry is not None else 0.0
        self.price_edit.setText(str(entry_edit))
        order_type = str(active_entry.get("order_type")) if active_entry else self._order_type(entry)
        self.order_type_label.setText(f"Order Type: {order_type}")
        if order_type in ("market", "limit", "stop market"):
            self.type_combo.setCurrentText(order_type)

        multiplier = _side_mult(plan_side)
        stop_valid = sl is not None and entry is not None and (sl < entry if multiplier > 0 else sl > entry)
        lots = None
        est_loss = None
        est_profit = None
        rr = None
        if bracket_active or submitted_pending_entry:
            lots = float(self._position.get("qty") or 0.0) if self._position is not None else float(active_entry.get("qty") or 0.0)
            if bracket_active and sl is not None and entry is not None:
                est_loss = abs(entry - sl) * lots
                if tp is not None and (tp > entry if multiplier > 0 else tp < entry):
                    est_profit = abs(tp - entry) * lots
                    rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) else None
        elif risk_target > 0 and stop_valid:
            per_lot_loss = abs(entry - sl)
            if per_lot_loss > 0:
                lots = risk_target / per_lot_loss
                est_loss = lots * per_lot_loss
                if tp is not None and (tp > entry if multiplier > 0 else tp < entry):
                    est_profit = abs(tp - entry) * lots
                    rr = abs(tp - entry) / per_lot_loss

        if active_entry:
            if not self._bracket_groups:
                self._bracket_groups = deepcopy(active_entry.get("bracket_groups") or [])
                self._group_plan_initialized = True
            if self._bracket_groups:
                if bracket_active:
                    if sl is not None:
                        self._bracket_groups[0]["sl"] = float(sl)
                    if tp is not None:
                        self._bracket_groups[0]["tp"] = float(tp)
                sl = float(self._bracket_groups[0].get("sl") or sl or 0.0)
                tp = float(self._bracket_groups[0].get("tp") or tp or 0.0)
                if entry is not None:
                    est_loss = sum(abs(entry - float(group.get("sl") or entry)) * float(group.get("qty") or 0.0) for group in self._bracket_groups)
                    est_profit = sum(abs(float(group.get("tp") or entry) - entry) * float(group.get("qty") or 0.0) for group in self._bracket_groups)
                    rr = est_profit / est_loss if est_loss > 1e-12 else None
        elif sl is not None and tp is not None and lots is not None and lots > 0:
            if not self._group_plan_initialized:
                self._bracket_groups = [{
                    "id": f"group-{self._next_bracket_group_number}",
                    "label": f"Group {self._next_bracket_group_number}",
                    "sl": float(sl), "tp": float(tp), "qty": float(lots),
                }]
                self._next_bracket_group_number += 1
                self._group_plan_initialized = True
            elif self._bracket_groups:
                self._bracket_groups[0]["sl"] = float(sl)
                self._bracket_groups[0]["tp"] = float(tp)
                if not self._manual_group_allocation:
                    allocated_before = sum(float(group.get("qty") or 0.0) for group in self._bracket_groups)
                    if allocated_before > 1e-12:
                        scale = float(lots) / allocated_before
                        for group in self._bracket_groups:
                            group["qty"] = float(group.get("qty") or 0.0) * scale
                    elif self._bracket_groups:
                        self._bracket_groups[0]["qty"] = float(lots)
            self._last_plan_lots = float(lots)

        self.qty_spin.setValue(0.0 if lots is None else lots)
        self.metric_labels["Est Loss"].setText("--" if est_loss is None else f"{est_loss:,.2f} USD")
        self.metric_labels["Est Profit"].setText("--" if est_profit is None else f"{est_profit:,.2f} USD")
        self.metric_labels["RR"].setText("--" if rr is None else f"{rr:.2f}")
        self.metric_labels["Risk Target"].setText(f"{risk_target:,.2f} USD" if risk_target > 0 else "--")
        self.metric_labels["Lots"].setText("--" if lots is None else f"{lots:.6f}")
        self._refresh_bracket_group_table(None if lots is None else float(lots))
        self._set_plan_controls_locked(bool(active_entry), submitted_pending_entry)
        groups_dirty = bool(
            bracket_active
            and self._confirmed_bracket_groups is not None
            and self._bracket_group_signature(self._bracket_groups)
            != self._bracket_group_signature(self._confirmed_bracket_groups)
        )

        self.order_plan_changed.emit({
            "mode": "market" if self.mode_market.isChecked() else "pending",
            "side": plan_side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "order_type": order_type,
            "lots": lots,
            "est_loss": est_loss,
            "est_profit": est_profit,
            "rr": rr,
            "bracket_groups": deepcopy(self._bracket_groups),
            "bracket_edit_enabled": bracket_active,
            "entry_locked": bool(active_entry),
            "plan_locked": submitted_pending_entry,
            "bracket_dirty": bool(
                bracket_active
                and (groups_dirty or (
                    self._confirmed_bracket is not None
                    and sl is not None
                    and tp is not None
                    and (abs(sl - self._confirmed_bracket[0]) > 1e-9 or abs(tp - self._confirmed_bracket[1]) > 1e-9)
                ))
            ),
        })

    def _set_plan_controls_locked(self, submitted: bool, waiting_entry: bool) -> None:
        for widget in (
            self.mode_pending, self.mode_market, self.btn_short, self.btn_long,
            self.equity_spin, self.risk_mode_button, self.risk_input,
            self.sl_edit, self.tp_edit,
        ):
            widget.setEnabled(not submitted)
        self.entry_edit.setEnabled(not submitted and not self.mode_market.isChecked())
        self.btn_add_bracket_group.setEnabled(not waiting_entry)
        self.btn_remove_bracket_group.setEnabled(not waiting_entry and bool(self._bracket_groups))

    def set_plan_price_from_chart(self, field: str, price: float) -> None:
        if price <= 0:
            return
        if field == "entry":
            if self._active_entry_id is not None or self.mode_market.isChecked():
                return
            self.entry_edit.setText(f"{float(price):.{self._price_precision(price)}f}")
            return
        if self._active_entry_id is not None and not self._active_bracket_is_live():
            return
        if field in ("sl", "tp") and self._bracket_groups:
            group = self._bracket_groups[0]
            group[field] = float(price)
            widget = self.sl_edit if field == "sl" else self.tp_edit
            widget.setText(f"{float(price):.{self._price_precision(price)}f}")
            self._refresh_bracket_group_table(float(self.qty_spin.value()))
            return
        if ":" in field:
            leg, group_id = field.split(":", 1)
            if leg not in ("sl", "tp"):
                return
            for group in self._bracket_groups:
                if group.get("id") == group_id:
                    group[leg] = float(price)
                    if self._bracket_groups.index(group) == 0:
                        widget = self.sl_edit if leg == "sl" else self.tp_edit
                        widget.setText(f"{float(price):.{self._price_precision(price)}f}")
                    else:
                        self.update_metrics()
                    self._refresh_bracket_group_table(float(self.qty_spin.value()))
                    return

    def set_context(self, case, raw_df: pd.DataFrame, replay):
        self.case = case
        self.raw_df = raw_df
        self.replay = replay
        self._active_entry_id = None
        self._confirmed_bracket = None
        self._confirmed_bracket_groups = None
        self._bracket_groups = []
        self._group_plan_initialized = False
        self._manual_group_allocation = False
        self._next_bracket_group_number = 1
        for widget in (self.entry_edit, self.sl_edit, self.tp_edit):
            widget.setText("0")
        # Unfilled orders are not research records and are never restored.
        self.pending_orders.clear()
        # Compatibility cleanup for v0.1.3 cases that may contain open/cancelled rows.
        if self.case is not None:
            filled_only = [o for o in self.case.orders if o.get("status") == "filled"]
            if len(filled_only) != len(self.case.orders):
                self.case.orders = filled_only
                self.case.touch()
                self.changed.emit()
        self._restore_active_bracket()
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
        if self._active_entry_id is not None:
            self._finish_active_bracket(self._active_entry_id, "closed")
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
        """Advance entries and their linked protection orders through newly revealed bars."""
        if self.case is None or self.raw_df.empty or current_ts <= previous_ts:
            self.refresh()
            return
        self._restore_active_bracket()
        bars = self.raw_df[(self.raw_df["timestamp"] > previous_ts) & (self.raw_df["timestamp"] <= current_ts)]
        changed = False
        closed_group_keys: set[tuple[str, str]] = set()
        closed_parent_ids: set[str] = set()
        for _, bar in bars.iterrows():
            bar_ts = float(bar["timestamp"])
            low = float(bar["low"]); high = float(bar["high"])
            # Entry fills happen first; attached protection cannot trigger on the same
            # OHLC bar because its intrabar event order is unknowable.
            for record in list(self.pending_orders):
                if record.get("role"):
                    continue
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
                    if record.get("bracket_order_ids"):
                        self._set_active_bracket(record, True, active_ts=bar_ts)
                    changed = True
            for protection in list(self.pending_orders):
                if protection not in self.pending_orders:
                    continue
                role = protection.get("role")
                if role not in ("stop_loss", "take_profit") or not protection.get("armed"):
                    continue
                parent_id = protection.get("parent_order_id")
                group_id = str(protection.get("group_id") or "")
                group_key = (str(parent_id), group_id)
                if group_key in closed_group_keys:
                    continue
                active_from = protection.get("active_from_ts")
                if active_from is not None and bar_ts <= float(active_from):
                    continue
                price = protection.get("requested_price")
                parent = self._find_entry_by_id(parent_id)
                if price is None or parent is None:
                    continue
                price = float(price)
                is_long = parent.get("side") == "long"
                if role == "stop_loss":
                    touched = low <= price if is_long else high >= price
                else:
                    touched = high >= price if is_long else low <= price
                if not touched:
                    continue

                protection["side"] = "short" if is_long else "long"
                protection["order_type"] = "stop loss" if role == "stop_loss" else "take profit"
                protection["origin"] = f"bracket-{role}"
                self._fill_record(protection, price, bar_ts)
                self.pending_orders.remove(protection)
                self.case.orders.append(protection)
                closed_group_keys.add(group_key)
                self._complete_protection_group(parent_id, group_id, protection)
                if self._position is None:
                    closed_parent_ids.add(parent_id)
                changed = True
        if changed:
            self.case.touch()
            self.changed.emit(); self.fills_changed.emit()
            if closed_parent_ids:
                self.entry_edit.setText("0")
                self.sl_edit.setText("0")
                self.tp_edit.setText("0")
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
                if record.get("role"):
                    parent_id = record.get("parent_order_id")
                    self._remove_bracket_group(parent_id, record.get("group_id"))
                    self._set_send_status("SL / TP group cancelled; other groups remain active", "#f5a623")
                    if self.case is not None:
                        self.case.touch()
                        self.changed.emit()
                else:
                    self._remove_bracket_orders(record.get("id"))
                    if self._active_entry_id == record.get("id"):
                        self._active_entry_id = None
                        self._confirmed_bracket = None
                    if record in self.pending_orders:
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
        if self._active_entry_id == order_id:
            self._finish_active_bracket(order_id, "cancelled")
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
        self.pending_orders.clear()
        self._active_entry_id = None
        self._confirmed_bracket = None
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
            role = o.get("role")
            type_text = o.get("order_type", "")
            status_text = o.get("status", "")
            if role == "stop_loss":
                type_text = f"SL stop market ({str(o.get('parent_order_id', ''))[-6:]})"
                status_text = "ACTIVE" if o.get("armed") else "WAIT ENTRY"
            elif role == "take_profit":
                type_text = f"TP limit ({str(o.get('parent_order_id', ''))[-6:]})"
                status_text = "ACTIVE" if o.get("armed") else "WAIT ENTRY"
            vals = [
                str(o.get("id", ""))[-6:], o.get("side", ""), type_text,
                "-" if o.get("requested_price") is None else f"{float(o['requested_price']):.{self._price_precision(o.get('requested_price'))}f}",
                f"{float(o.get('qty') or 0):.4f}", status_text,
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
