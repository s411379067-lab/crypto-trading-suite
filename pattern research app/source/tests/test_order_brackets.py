import pytest
import pandas as pd
from pyqtgraph.Qt import QtWidgets

from pattern_analyzer.order_panel import OrderPanel
from shared_core.order_brackets import (
    available_bracket_qty,
    create_pending_bracket,
    move_bracket_entry,
    move_bracket_leg,
    pending_bracket_order_specs,
    set_bracket_group_qty,
    set_bracket_qty,
)


def test_initial_short_bracket_has_symmetric_stop_and_target():
    bracket = create_pending_bracket("short", 100.0, 2.0)

    assert bracket["state"] == "pending"
    assert bracket["groups"][0]["qty"] == 2.0
    assert bracket["groups"][0]["stop_price"] == 101.0
    assert bracket["groups"][0]["target_price"] == 99.0


def test_initial_long_bracket_has_symmetric_stop_and_target():
    bracket = create_pending_bracket("long", 100.0, 1.0)

    assert bracket["groups"][0]["stop_price"] == 99.0
    assert bracket["groups"][0]["target_price"] == 101.0


def test_initial_bracket_requires_positive_quantity():
    with pytest.raises(ValueError, match="positive"):
        create_pending_bracket("short", 100.0, 0)


def test_moving_entry_shifts_its_stop_and_target_together():
    bracket = create_pending_bracket("short", 100.0, 1.0)
    moved = move_bracket_entry(bracket, 105.0)

    assert bracket["entry_price"] == 100.0
    assert moved["entry_price"] == 105.0
    assert moved["groups"][0]["stop_price"] == 106.0
    assert moved["groups"][0]["target_price"] == 104.0


def test_moving_one_leg_keeps_the_other_leg_unchanged():
    bracket = create_pending_bracket("long", 100.0, 1.0)
    group_id = bracket["groups"][0]["id"]
    moved = move_bracket_leg(bracket, group_id, "stop", 97.0)

    assert moved["groups"][0]["stop_price"] == 97.0
    assert moved["groups"][0]["target_price"] == 101.0


def test_reducing_entry_lots_is_rejected_when_it_would_over_allocate_groups():
    bracket = create_pending_bracket("short", 100.0, 2.0)

    with pytest.raises(ValueError, match="allocated"):
        set_bracket_qty(bracket, 1.0)


def test_entry_lots_can_increase_without_changing_existing_group_lots():
    bracket = create_pending_bracket("long", 100.0, 1.0)
    edited = set_bracket_qty(bracket, 2.5)

    assert edited["groups"][0]["qty"] == 1.0
    assert available_bracket_qty(edited) == 1.5


def test_sl_and_tp_share_one_editable_group_lot_value():
    bracket = create_pending_bracket("short", 100.0, 2.0)
    group_id = bracket["groups"][0]["id"]
    edited = set_bracket_group_qty(bracket, group_id, 0.75)

    assert edited["groups"][0]["qty"] == 0.75
    assert available_bracket_qty(edited) == 1.25


def test_group_lots_cannot_exceed_entry_lots():
    bracket = create_pending_bracket("long", 100.0, 1.0)
    group_id = bracket["groups"][0]["id"]

    with pytest.raises(ValueError, match="exceeds"):
        set_bracket_group_qty(bracket, group_id, 1.5)


def test_pending_submit_specs_include_entry_stop_and_target_with_linked_lots():
    bracket = create_pending_bracket("short", 100.0, 2.0)
    bracket["groups"][0]["qty"] = 0.75
    specs = pending_bracket_order_specs(bracket)

    assert [(spec["role"], spec["side"], spec["order_type"], spec["qty"]) for spec in specs] == [
        ("entry", "short", "limit", 2.0),
        ("stop", "long", "stop market", 0.75),
        ("target", "long", "limit", 0.75),
    ]


def test_pending_submit_keeps_brackets_linked_until_entry_then_oco_exits():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class Case:
        orders = []
        display = {"timezone": "UTC"}

        def touch(self):
            pass

    class Replay:
        current_ts = 0.0

    panel = OrderPanel()
    panel.case = Case()
    panel.raw_df = pd.DataFrame([
        {"timestamp": 60.0, "low": 99.0, "high": 101.0, "close": 100.0},
        {"timestamp": 120.0, "low": 98.0, "high": 99.0, "close": 99.0},
    ])
    panel.replay = Replay()
    panel.pending_bracket = create_pending_bracket("short", 100.0, 1.0)

    panel.submit_pending_bracket()
    assert [record["bracket_role"] for record in panel.pending_orders] == ["entry", "stop", "target"]

    panel.process_replay_advance(0.0, 60.0)
    assert [record["bracket_role"] for record in panel.case.orders] == ["entry"]
    assert {record["bracket_role"] for record in panel.pending_orders} == {"stop", "target"}

    panel.process_replay_advance(60.0, 120.0)
    assert [record["bracket_role"] for record in panel.case.orders] == ["entry", "target"]
    assert panel.pending_orders == []


def test_manual_market_close_cancels_submitted_bracket_protection():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class Case:
        orders = []
        display = {"timezone": "UTC"}

        def touch(self):
            pass

    class Replay:
        current_ts = 0.0

    panel = OrderPanel()
    panel.case = Case()
    panel.raw_df = pd.DataFrame([{"timestamp": 60.0, "low": 99.0, "high": 101.0, "close": 100.0}])
    panel.replay = Replay()
    panel.pending_bracket = create_pending_bracket("short", 100.0, 1.0)
    panel.submit_pending_bracket()
    panel.process_replay_advance(0.0, 60.0)
    panel.replay.current_ts = 60.0

    panel.close_position()

    assert panel.pending_orders == []
    assert panel.submitted_bracket is None
